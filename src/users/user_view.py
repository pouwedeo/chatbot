import streamlit as st
from src.model.chat_bot import get_chroma_db, get_retriever, ChatbotWithMemory
from src.users.feedback import save_feedback


def get_data():
    
   
    vector_store = get_chroma_db("src/docs/chroma_db/")
    
    # Configuration du retriever
    retriever = get_retriever(vector_store)

    # Création de la chaîne de chatbot
    st.session_state.chatbot_chain = ChatbotWithMemory(retriever)


def feedback_buttons(message_id):

    icons = {
        "Copy": ":material/content_copy:",
        "Like": ":material/thumb_up:",
        "Dislike": ":material/thumb_down:",
        "Download": ":material/cloud_download:",
    }
    cols = st.columns(20)

    for col, (label, icon) in zip(cols, icons.items()):

        current_feedback = st.session_state.messages[int(
            message_id)]["feedback"]
        is_active = current_feedback.get(label, False)

        # Style différent pour Like/Dislike
        btn_type = "secondary" if is_active else "tertiary"

        if col.button(
            icon,
            key=f"fb_{message_id}_{label}",  # Clé unique
            type=btn_type,
            use_container_width=True,
            help=label  # Info-bulle
        ):
            # Logique de basculement
            if label in ["Like", "Dislike"]:
                current_feedback[label] = not is_active
                # Exclusion mutuelle Like/Dislike
                if label == "Like" and current_feedback["Like"]:
                    current_feedback["Dislike"] = False
                elif label == "Dislike" and current_feedback["Dislike"]:
                    current_feedback["Like"] = False
            else:
                # Basculer simplement pour Copy/Download
                current_feedback[label] = not is_active

            save_feedback(label, message_id)
            st.rerun()


def user_page():
    # Get data from chroma
    get_data()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Affichage de l'historique des messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                feedback_buttons(message["id"])

    # Gestion de l'entrée utilisateur
    if prompt := st.chat_input("Posez votre question..."):
        # Ajout du message utilisateur à l'historique

        # Affichage du message utilisateur
        with st.chat_message("user"):
            col1, col2 = st.columns(2)
            col1.container()
            col2.info(prompt)

        user_message = {
            "role": "user",
            "content": prompt,
            "id": str(len(st.session_state.messages))  # ID unique
        }
        st.session_state.messages.append(user_message)

        # Vérification que le chatbot est initialisé
        if "chatbot_chain" not in st.session_state:
            st.warning("Veuillez d'abord charger des données.")
            st.stop()

        # Génération de la réponse
        with st.chat_message("assistant"):
            with st.spinner("Je réfléchis..."):
                try:
                    response = st.session_state.chatbot_chain.invoke(prompt)
                    assistant_message = {
                        "role": "assistant",
                        "content": response,
                        "id": str(len(st.session_state.messages)),
                        "feedback": {
                            "Copy": False,
                            "Like": False,
                            "Dislike": False,
                            "Download": False
                        }
                    }
                    st.session_state.messages.append(assistant_message)
                    st.rerun()
                except Exception as e:
                    st.error(f"Une erreur est survenue: {e}")
