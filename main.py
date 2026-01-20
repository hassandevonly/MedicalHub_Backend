"""
API FastAPI pour le diagnostic médical assisté par IA.

Cette API expose un endpoint POST /predict pour la détection de pneumonie
à partir d'images radiographiques pulmonaires en utilisant un modèle CNN
pré-entraîné.

Auteur: Projet académique Master IASD
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
import json

# Import des modules de prédiction
from utils.model_loader import load_model, DECISION_THRESHOLD
from utils.predictor import predict_from_bytes

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialisation de l'application FastAPI
app = FastAPI(
    title="Medical AI Backend - Pneumonia Detection",
    version="1.0.0",
    description="API de diagnostic médical assisté par IA pour la détection de pneumonie à partir de radiographies pulmonaires"
)

# Configuration CORS pour permettre les requêtes depuis le frontend Angular
# Permet la communication entre le frontend et le backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4200",  # Frontend Angular en développement
        "http://127.0.0.1:4200",  # Alternative localhost
        "http://localhost:3000",  # Autres ports possibles
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Méthodes HTTP autorisées
    allow_headers=[
        "Content-Type",
        "Authorization",
        "Accept",
        "Origin",
        "X-Requested-With",
    ],
    expose_headers=["*"],  # Expose tous les headers dans la réponse
)


# Modèle de réponse Pydantic strict et scientifique
class PredictionResponse(BaseModel):
    """
    Modèle de réponse pour la prédiction de pneumonie.
    
    Format strict et scientifique conforme aux standards de recherche médicale.
    """
    prediction: str = Field(
        ...,
        description="Classe prédite: 'PNEUMONIA' ou 'NORMAL'",
        examples=["PNEUMONIA", "NORMAL"]
    )
    probability: float = Field(
        ...,
        description="Probabilité brute du modèle (valeur entre 0 et 1). "
                   "Représente P(pneumonia). Plus la valeur est proche de 1, "
                   "plus le modèle est confiant que l'image présente une pneumonie.",
        ge=0.0,
        le=1.0,
        examples=[0.8235, 0.1234]
    )
    decision_threshold: float = Field(
        ...,
        description="Seuil de décision utilisé pour la classification binaire. "
                   "Standard: 0.5. Si probability >= threshold → PNEUMONIA, sinon → NORMAL.",
        examples=[0.5]
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "PNEUMONIA",
                "probability": 0.8235,
                "decision_threshold": 0.5
            }
        }


@app.on_event("startup")
async def startup_event():
    """
    Événement de démarrage de l'application.
    
    Charge le modèle CNN au démarrage pour optimiser les performances.
    Le modèle est chargé une seule fois en mémoire et réutilisé pour toutes les prédictions.
    """
    try:
        logger.info("=" * 60)
        logger.info("DÉMARRAGE DE L'APPLICATION")
        logger.info("=" * 60)
        
        # Charger le modèle au démarrage
        model = load_model()
        
        logger.info("Application prête à recevoir des requêtes")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"ERREUR CRITIQUE AU DÉMARRAGE: {str(e)}")
        logger.error("L'application ne peut pas démarrer sans le modèle")
        raise


@app.get("/")
def root():
    """
    Endpoint de santé pour vérifier que l'API fonctionne.
    Utile pour le frontend pour vérifier la connectivité.
    """
    return {
        "status": "operational",
        "service": "Pneumonia Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/",
            "predict": "/predict",
            "docs": "/docs"
        }
    }


@app.get("/health")
def health_check():
    """
    Endpoint de santé détaillé pour le monitoring.
    Retourne l'état du service et du modèle.
    """
    try:
        from utils.model_loader import get_model
        model = get_model()
        model_status = "loaded" if model is not None else "not_loaded"
    except Exception as e:
        model_status = f"error: {str(e)}"
    
    return {
        "status": "operational",
        "model_status": model_status,
        "service": "Pneumonia Detection API",
        "version": "1.0.0"
    }


@app.get("/test-response")
def test_response():
    """
    Endpoint de test pour vérifier la structure de la réponse.
    Retourne un exemple de réponse PredictionResponse.
    """
    example_response = PredictionResponse(
        prediction="NORMAL",
        probability=0.1234,
        decision_threshold=0.5
    )
    return example_response.model_dump()


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(..., description="Image radiographique pulmonaire à analyser")):
    """
    Endpoint de prédiction pour la détection de pneumonie.
    
    Reçoit une image radiographique pulmonaire et retourne une prédiction
    binaire (PNEUMONIA ou NORMAL) avec la probabilité associée.
    
    Pipeline de traitement:
    1. Réception de l'image via multipart/form-data
    2. Prétraitement (redimensionnement 150x150, normalisation [0,1])
    3. Prédiction avec le modèle CNN
    4. Application de la règle de décision (seuil = 0.5)
    5. Retour de la réponse JSON
    
    Args:
        file: Fichier image uploadé (format supporté: JPG, PNG, etc.)
        
    Returns:
        PredictionResponse: Réponse JSON contenant:
            - prediction: "PNEUMONIA" ou "NORMAL"
            - probability: Probabilité brute entre 0 et 1
            - decision_threshold: Seuil utilisé (0.5)
            
    Raises:
        HTTPException 400: Si l'image est invalide ou ne peut pas être traitée
        HTTPException 500: Si une erreur serveur survient
    """
    try:
        # Log de la requête
        logger.info("=" * 60)
        logger.info("NOUVELLE REQUÊTE POST /predict")
        logger.info(f"Fichier: {file.filename}")
        logger.info(f"Type MIME: {file.content_type}")
        
        # Lire les bytes de l'image
        image_bytes = await file.read()
        logger.info(f"Taille: {len(image_bytes)} bytes ({len(image_bytes) / 1024:.2f} KB)")
        
        # Effectuer la prédiction
        logger.info("Traitement de l'image...")
        result = predict_from_bytes(image_bytes)
        
        # Log de la réponse brute
        logger.info("-" * 60)
        logger.info("RÉPONSE BRUTE (dict):")
        logger.info(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Créer l'objet Pydantic et vérifier la sérialisation
        response_obj = PredictionResponse(**result)
        response_dict = response_obj.model_dump()
        
        logger.info("RÉPONSE SÉRIALISÉE (JSON):")
        logger.info(json.dumps(response_dict, indent=2, ensure_ascii=False))
        logger.info("=" * 60)
        
        return response_obj
        
    except ValueError as e:
        # Erreur de validation (image invalide, format incorrect, etc.)
        logger.error(f"ERREUR DE VALIDATION: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Erreur de traitement de l'image: {str(e)}"
        )
    except Exception as e:
        # Autre erreur serveur
        logger.error(f"ERREUR SERVEUR: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur interne du serveur: {str(e)}"
        )
