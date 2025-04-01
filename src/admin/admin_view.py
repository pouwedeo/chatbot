import streamlit as st
from src.model.chat_bot import load_pdf, split_documents, get_embedding_model, create_vector_store


def upload_file():
    # upload file
    uploaded_files = st.file_uploader(
        "Choisir le fichier", accept_multiple_files=True)
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.spinner("Chargement et traitement des données..."):
                try:
                    
                    # Chargement et traitement des documents
                    documents = load_pdf(uploaded_file)
                    
                    # Vérification du type de documents
                    if not isinstance(documents, list) or not all(hasattr(doc, "page_content") for doc in documents):
                        raise ValueError("Les documents ne sont pas correctement formatés.")
                    splitted_docs = split_documents(documents)
                    
                    # Création de la base vectorielle
                    embedding_model = get_embedding_model()
                    create_vector_store(splitted_docs, embedding_model)
                   
                    st.success("Données chargées avec succès !")
                except Exception as e:
                    st.error(f"Erreur lors du chargement des données: {e}")
  
                    
