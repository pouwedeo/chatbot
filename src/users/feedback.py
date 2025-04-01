import os
import pandas as pd
import csv

# Chemin du fichier CSV
FEEDBACK_FILE = "src/admin/feedback.csv"

# Initialisation du fichier CSV avec en-têtes
if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, mode="w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Copy", "Like",
                        "Dislike", "Download", "MessageID"])


def load_feedback():
    """Charge les données de feedback existantes"""
    try:
        return pd.read_csv(FEEDBACK_FILE)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame(columns=["Timestamp", "Copy", "Like", "Dislike", "Download", "MessageID"])


def save_feedback(action, message_id=None):
    """
    Enregistre une action de feedback dans le CSV
    Args:
        action (str): Un parmi "Copy", "Like", "Dislike", "Download"
        message_id (str): ID unique du message associé
    """
    # Crée un nouveau dictionnaire avec toutes les colonnes
    new_entry = {
        "Timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Copy": 0,
        "Like": 0,
        "Dislike": 0,
        "Download": 0,
        "MessageID": message_id
    }

    # Active uniquement l'action correspondante
    if action in new_entry:
        new_entry[action] = 1

    # Charge, met à jour et sauvegarde
    feedback_data = load_feedback()
    feedback_data = pd.concat(
        [feedback_data, pd.DataFrame([new_entry])], ignore_index=True)

    # Sauvegarde avec vérification du répertoire
    os.makedirs(os.path.dirname(FEEDBACK_FILE), exist_ok=True)
    feedback_data.to_csv(FEEDBACK_FILE, index=False, encoding="utf-8")
