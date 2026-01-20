"""
Module de prédiction pour la détection de pneumonie.

Ce module contient la logique de prédiction utilisant le modèle CNN chargé.
Il effectue le prétraitement de l'image et applique la règle de décision
basée sur un seuil de probabilité.
"""

from utils.model_loader import get_model, INPUT_SIZE, DECISION_THRESHOLD
from utils.image_preprocessing import preprocess_image
import logging

logger = logging.getLogger(__name__)


def predict_from_bytes(image_bytes: bytes) -> dict:
    """
    Effectue une prédiction de pneumonie à partir des bytes d'une image.
    
    Pipeline de prédiction:
    1. Prétraitement de l'image (redimensionnement, normalisation, batch)
    2. Prédiction avec le modèle CNN
    3. Extraction de la probabilité brute (valeur entre 0 et 1)
    4. Application de la règle de décision avec seuil = 0.5
    
    Règle de décision:
    - prob >= 0.5 → PNEUMONIA (classe positive)
    - prob < 0.5 → NORMAL (classe négative)
    
    Le seuil de 0.5 est standard pour les problèmes de classification binaire
    et équilibre la sensibilité et la spécificité du modèle.
    
    Args:
        image_bytes: Bytes de l'image radiographique à analyser
        
    Returns:
        dict: Dictionnaire contenant:
            - prediction: "PNEUMONIA" ou "NORMAL"
            - probability: Probabilité brute entre 0 et 1 (probabilité de pneumonie)
            - decision_threshold: Seuil utilisé (0.5)
            
    Raises:
        ValueError: Si l'image ne peut pas être traitée
        Exception: Si la prédiction échoue
    """
    try:
        # 1. Prétraitement de l'image
        # - Conversion en RGB
        # - Redimensionnement à 150x150 (taille d'entrée du modèle)
        # - Normalisation des pixels dans [0, 1] (division par 255.0)
        # - Ajout de la dimension batch
        logger.info("Prétraitement de l'image...")
        preprocessed_image = preprocess_image(image_bytes, target_size=INPUT_SIZE)
        
        # Vérification de la forme attendue: (1, 150, 150, 3)
        if preprocessed_image.shape != (1, INPUT_SIZE[0], INPUT_SIZE[1], 3):
            raise ValueError(
                f"Forme d'image incorrecte: {preprocessed_image.shape}. "
                f"Attendu: (1, {INPUT_SIZE[0]}, {INPUT_SIZE[1]}, 3)"
            )
        
        # 2. Prédiction avec le modèle CNN
        logger.info("Exécution de la prédiction avec le modèle CNN...")
        model = get_model()
        
        # model.predict retourne un array de forme (1, 1) pour un modèle binaire
        # La valeur [0][0] est la probabilité P(pneumonia)
        prediction_output = model.predict(preprocessed_image, verbose=0)
        
        # Log détaillé de la sortie brute du modèle
        logger.info(f"Sortie brute du modèle (shape: {prediction_output.shape}): {prediction_output}")
        logger.info(f"Valeur [0][0]: {prediction_output[0][0]}")
        
        # 3. Extraction de la probabilité brute
        # Le modèle retourne la probabilité de la classe positive (PNEUMONIA)
        # Valeur entre 0 et 1, où:
        # - 0 = très confiant que c'est NORMAL
        # - 1 = très confiant que c'est PNEUMONIA
        probability = float(prediction_output[0][0])
        
        # Log de la probabilité extraite
        logger.info(f"Probabilité extraite: {probability:.6f}")
        
        # Vérification que la probabilité est dans l'intervalle valide
        if not (0.0 <= probability <= 1.0):
            raise ValueError(
                f"Probabilité invalide: {probability}. "
                "Doit être entre 0 et 1."
            )
        
        # 4. Application de la règle de décision
        # Seuil de 0.5: standard pour la classification binaire
        # - prob >= 0.5 → PNEUMONIA (classe positive)
        # - prob < 0.5 → NORMAL (classe négative)
        if probability >= DECISION_THRESHOLD:
            prediction = "PNEUMONIA"
        else:
            prediction = "NORMAL"
        
        logger.info(
            f"RÉSULTAT FINAL - Prédiction: {prediction} "
            f"(probabilité brute: {probability:.6f}, seuil: {DECISION_THRESHOLD})"
        )
        logger.info(
            f"Interprétation: probabilité {probability:.4f} "
            f"{'>= seuil' if probability >= DECISION_THRESHOLD else '< seuil'} "
            f"→ {prediction}"
        )
        
        # 5. Construction de la réponse
        return {
            "prediction": prediction,
            "probability": round(probability, 4),  # Arrondi à 4 décimales pour précision
            "decision_threshold": DECISION_THRESHOLD
        }
        
    except ValueError as e:
        logger.error(f"Erreur de validation: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}", exc_info=True)
        raise Exception(f"Erreur de prédiction: {str(e)}")
