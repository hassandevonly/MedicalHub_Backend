"""
Module de prétraitement d'images pour le diagnostic médical assisté par IA.

Ce module contient les fonctions nécessaires pour prétraiter les images radiographiques
avant leur passage dans un modèle de deep learning.
"""

import numpy as np
from PIL import Image
from io import BytesIO
from typing import Tuple


def read_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Lit une image à partir de bytes.
    
    Args:
        image_bytes: Bytes de l'image à lire
        
    Returns:
        Image PIL ouverte
        
    Raises:
        ValueError: Si l'image ne peut pas être lue
    """
    try:
        image = Image.open(BytesIO(image_bytes))
        return image
    except Exception as e:
        raise ValueError(f"Impossible de lire l'image: {str(e)}")


def convert_to_rgb(image: Image.Image) -> Image.Image:
    """
    Convertit une image en mode RGB.
    
    Args:
        image: Image PIL à convertir
        
    Returns:
        Image PIL en mode RGB
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


def resize_image(image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """
    Redimensionne une image à la taille cible.
    
    Args:
        image: Image PIL à redimensionner
        target_size: Tuple (largeur, hauteur) de la taille cible (par défaut: 224x224)
        
    Returns:
        Image PIL redimensionnée
    """
    return image.resize(target_size, Image.Resampling.LANCZOS)


def normalize_pixels(image_array: np.ndarray) -> np.ndarray:
    """
    Normalise les pixels de l'image dans l'intervalle [0, 1].
    
    Args:
        image_array: Tableau NumPy de l'image (valeurs 0-255)
        
    Returns:
        Tableau NumPy normalisé (valeurs 0-1)
    """
    # Convertir en float32 pour éviter les problèmes de précision
    normalized = image_array.astype(np.float32) / 255.0
    return normalized


def add_batch_dimension(image_array: np.ndarray) -> np.ndarray:
    """
    Ajoute la dimension batch pour un modèle CNN.
    
    Args:
        image_array: Tableau NumPy de forme (height, width, channels)
        
    Returns:
        Tableau NumPy de forme (1, height, width, channels)
    """
    return np.expand_dims(image_array, axis=0)


def preprocess_image(image_bytes: bytes, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Pipeline complet de prétraitement d'image pour un modèle CNN.
    
    Étapes:
    1. Lire l'image à partir des bytes
    2. Convertir en RGB
    3. Redimensionner à la taille cible
    4. Convertir en tableau NumPy
    5. Normaliser les pixels dans [0, 1]
    6. Ajouter la dimension batch
    
    Args:
        image_bytes: Bytes de l'image à prétraiter
        target_size: Tuple (largeur, hauteur) de la taille cible (par défaut: 224x224)
        
    Returns:
        Tableau NumPy prétraité de forme (1, height, width, 3) avec valeurs normalisées [0, 1]
        
    Raises:
        ValueError: Si l'image ne peut pas être traitée
    """
    # 1. Lire l'image
    image = read_image_from_bytes(image_bytes)
    
    # 2. Convertir en RGB
    image = convert_to_rgb(image)
    
    # 3. Redimensionner
    image = resize_image(image, target_size)
    
    # 4. Convertir en tableau NumPy (shape: height, width, 3)
    image_array = np.array(image)
    
    # 5. Normaliser les pixels
    image_array = normalize_pixels(image_array)
    
    # 6. Ajouter la dimension batch (shape: 1, height, width, 3)
    image_array = add_batch_dimension(image_array)
    
    return image_array

