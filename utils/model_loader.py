"""
Module de chargement du modèle CNN pour la détection de pneumonie.

Ce module charge le modèle Keras pré-entraîné au démarrage de l'application
pour éviter de recharger le modèle à chaque prédiction, optimisant ainsi
les performances de l'API.
"""

import tensorflow as tf
import logging
import os

# Configuration du logging
logger = logging.getLogger(__name__)

# Chemin vers le modèle pré-entraîné
MODEL_PATH = "model/pneumonia_model.h5"

# Taille d'entrée attendue par le modèle (hauteur, largeur)
# Le modèle a été entraîné avec des images de 150x150 pixels
INPUT_SIZE = (150, 150)

# Mapping des classes du modèle
# Le modèle est binaire : 0 = NORMAL, 1 = PNEUMONIA
# La sortie du modèle est une probabilité P(pneumonia)
CLASS_MAPPING = {0: "NORMAL", 1: "PNEUMONIA"}

# Seuil de décision pour la classification binaire
# prob >= DECISION_THRESHOLD → PNEUMONIA
# prob < DECISION_THRESHOLD → NORMAL
DECISION_THRESHOLD = 0.5

# Variable globale pour stocker le modèle chargé
model = None


def load_model():
    """
    Charge le modèle Keras pré-entraîné depuis le fichier .h5.
    
    Cette fonction doit être appelée au démarrage de l'application FastAPI
    pour charger le modèle une seule fois en mémoire.
    
    Returns:
        tf.keras.Model: Le modèle Keras chargé
        
    Raises:
        FileNotFoundError: Si le fichier modèle n'existe pas
        Exception: Si le chargement du modèle échoue
    """
    global model
    
    if model is not None:
        logger.info("Modèle déjà chargé en mémoire")
        return model
    
    # Vérifier que le fichier modèle existe
    if not os.path.exists(MODEL_PATH):
        error_msg = f"Fichier modèle introuvable: {MODEL_PATH}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        logger.info(f"Chargement du modèle depuis {MODEL_PATH}...")
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("Modèle chargé avec succès")
        
        # Afficher un résumé du modèle pour vérification
        logger.info(f"Taille d'entrée attendue: {INPUT_SIZE}")
        logger.info(f"Seuil de décision: {DECISION_THRESHOLD}")
        
        return model
    except Exception as e:
        error_msg = f"Erreur lors du chargement du modèle: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)


def get_model():
    """
    Retourne le modèle chargé. Charge le modèle si ce n'est pas déjà fait.
    
    Returns:
        tf.keras.Model: Le modèle Keras
    """
    global model
    if model is None:
        model = load_model()
    return model
