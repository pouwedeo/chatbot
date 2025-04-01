import streamlit as st 
import pandas as pd
import plotly.express as px 


def feedback_show():
    df = pd.read_csv("src/admin/feedback.csv")

    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Agréger les données par action
    summary = df[['Like', 'Dislike', 'Copy', 'Download']].sum().reset_index()
    summary.columns = ['Action', 'Count']

    # Créer le bar chart avec Plotly
    fig = px.bar(
        summary,
        x='Action',
        y='Count',
        color='Action',
        title="Feedback Utilisateurs",
        labels={'Count': 'Nombre de feedbacks', 'Action': 'Type de feedback'},
        color_discrete_map={
            'Like': '#4CAF50',  # Vert
            'Dislike': '#F44336',  # Rouge
            'Copy': '#2196F3',  # Bleu
            'Download': '#FFC107'  # Jaune
        }
    )

    # Personnalisation du graphique
    fig.update_layout(
        xaxis_title=None,
        yaxis_title="Nombre de feedbacks",
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)'
    )

    # Afficher le graphique dans Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Afficher les données brutes
