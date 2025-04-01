import os
import fitz
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from dotenv import load_dotenv
from together import Together
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableLambda
from langchain.memory import ConversationBufferWindowMemory


# Charger les variables d'environnement
load_dotenv()

# Configuration de l'API Together
together_api_key = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=together_api_key)


# Fonctions du chatbot (inchangées)


# Fonction pour extraire du texte des PDFs scannés avec
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
tessdata_dir_config = r'--tessdata-dir "C:\Program Files\Tesseract-OCR\tessdata"'


def extract_text_from_scanned_pdf(pdf_path):
    # Convertir le PDF en images
    images = convert_from_path(pdf_path)

    # Extraire le texte de chaque image
    text = "\n".join([pytesseract.image_to_string(img, lang="fra", config=tessdata_dir_config)
                     for img in images])

    return text


# Fonction pour extraire du texte des PDFs normaux


def extract_text_pymupdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text.strip()

# Fonction principale pour charger un PDF avec OCR si nécessaire


def load_pdf(uploaded_file):
    temp_pdf_path = "temp_uploaded.pdf"

    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extraction du texte
    text = extract_text_pymupdf(temp_pdf_path)
    if not text:
        text = extract_text_from_scanned_pdf(temp_pdf_path)

    os.remove(temp_pdf_path)  # Nettoyage du fichier temporaire

    # Retourner un objet Document pour compatibilité avec Langchain
    return [Document(page_content=text, metadata={"source": uploaded_file.name})]

# Découpage des documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len)
    return text_splitter.split_documents(documents)


def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )


def create_vector_store(documents, embedding_model,
                        persist_directory="src/docs/chroma_db/"):
    return Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory
    )


def get_chroma_db(db_directory):
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2")

    vector_store = Chroma(
        persist_directory=db_directory,
        embedding_function=embedding_model
    )

    return vector_store


def get_retriever(vector_store, k=3):
    return vector_store.as_retriever(search_kwargs={"k": k})
 

class TogetherLLM:
    def __init__(self, model="meta-llama/Llama-3-70b-chat-hf"):
        self.model = model
        self.client = Together(api_key=together_api_key)

    def __call__(self, prompt):
        # 🔹 Vérifie et extrait correctement le texte du prompt
        if hasattr(prompt, "to_string"):
            prompt = prompt.to_string()  # Convertir en texte brut si nécessaire
        elif isinstance(prompt, list):
            # Concaténer les messages
            prompt = "\n".join([msg.content for msg in prompt])

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.7
        )
        return response.choices[0].message.content


class ChatbotWithMemory:
    def __init__(self, retriever, k=3):
        self.memory = ConversationBufferWindowMemory(
            k=5,  # Conserve les 5 derniers échanges
            memory_key="chat_history",
            return_messages=True
        )
        self.retriever = retriever
        self.chain = self._build_chain()

    def _build_chain(self):
        template = """### Rôle:
            Vous êtes un expert assistant conversationnel français qui combine intelligemment :
            1. L'historique de la conversation
            2. Les documents de référence fournis
            3. Vos connaissances générales

            ### Instructions:
            - Répondez exclusivement en français
            - Soyez précis et structuré
            - Adaptez votre ton à la question
            - Vérifiez toujours les faits dans le contexte fourni
            - Si la réponse nécessite une longue explication, donnez d'abord une réponse concise puis détaillez
            - Adaptez votre conclusion pour engager la conversation
            
            ### Historique de la conversation:
            {chat_history}

            ### Contexte documentaire pertinent:
            {context}

            ### Question à analyser:
            {question}

            ### Format de réponse:
            1.  Maximum 4 phrases
            2. **Détails** (si nécessaire) : Explications complémentaires
            3.  Adaptée au contexte (voir exemples ci-dessous)
            4. [Une phrase ouverte encourageant l'utilisateur à poser des questions complémentaires]

            ### Expressions de fin à adapter:
            - Pour les questions factuelles: "Ces informations proviennent de nos documents officiels."
            - Pour les demandes complexes: "N'hésitez pas à me demander des précisions sur l'un de ces points."
            - Pour les sujets sensibles: "Je vous recommande de consulter notre service juridique pour une analyse personnalisée."
            - Pour les remerciements: "Je reste à votre disposition pour toute question complémentaire."
            - Par défaut: "Cette réponse s'appuie sur nos ressources documentaires actuelles."
            
            ###Exemple de conclusion :
            - "Souhaitez"-vous approfondir un point spécifique ?"
            - "Dois-je clarifier certains éléments ?"
            - "Avez-vous d'autres questions sur ce sujet ?"
            -"N'hésitez pas à me demander des précisions sur l'un de ces points ou toute autre question sur ce sujet."
            
            ### Réponse à construire:"""

        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        def load_memory(_):
            return self.memory.load_memory_variables({})["chat_history"]

        chain = (
            {
                "question": RunnablePassthrough(),
                "chat_history": RunnableLambda(load_memory),
                "context": lambda x: format_docs(self.retriever.invoke(x["question"]))
            }
            | prompt
            | TogetherLLM()
            | StrOutputParser()
        )
        return chain

    def invoke(self, question):
        response = self.chain.invoke({"question": question})
        self.memory.save_context(
            {"input": question},
            {"output": response}
        )
        return response