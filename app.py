import streamlit as st
from src.admin.admin_view import upload_file
from src.admin.footer import footer_page
from src.users.user_view import user_page
from src.admin.feedback_stat import feedback_show



# Side bar
user_mode = st.sidebar.radio(
    "Choisir le mode d'utilisation",
    ["User", "Admin"]
)

if user_mode == "User":
    # User page for questions
   
    user_page()
    
elif user_mode == "Admin":
    file_page, feedback_page = st.tabs(["Charger le fichiers", "Voir les feedbacks utilisateurs"])
    with file_page:
        # Uploade file for rag
        st.subheader("Charger les fichiers")
        upload_file()
    with feedback_page:
        st.subheader("Visualisation des feedbacks utilisateurs")
        feedback_show()


# Footer page
footer_page()




