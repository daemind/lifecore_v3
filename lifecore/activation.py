#!/usr/bin/env python3
"""
LifeCore V3 - Activations
=========================

Fonctions d'activation pour remplacer les seuils hardcodés.
Utilisation de fonctions continues (sigmoid, relu, softplus) au lieu de if/else.

Principes:
- Pas de seuils hardcodés
- Fonctions différentiables
- Transitions douces
"""

import numpy as np
from typing import Callable


# === FONCTIONS D'ACTIVATION ===

def sigmoid(x: float, center: float = 0.0, steepness: float = 1.0) -> float:
    """Sigmoid: transition douce entre 0 et 1.
    
    Args:
        x: Valeur d'entrée
        center: Point où sigmoid = 0.5
        steepness: Pente de la transition (plus grand = plus raide)
        
    Returns:
        Valeur entre 0 et 1
    """
    return 1.0 / (1.0 + np.exp(-steepness * (x - center)))


def relu(x: float) -> float:
    """ReLU: max(0, x)."""
    return float(np.maximum(0, x))


def softplus(x: float, beta: float = 1.0) -> float:
    """Softplus: version lisse de ReLU.
    
    softplus(x) ≈ relu(x) mais différentiable en 0.
    """
    return float(np.log1p(np.exp(beta * x)) / beta)


def tanh_scaled(x: float, scale: float = 1.0) -> float:
    """Tanh scalé: sortie entre -scale et +scale."""
    return float(np.tanh(x) * scale)


def smooth_threshold(x: float, threshold: float, steepness: float = 5.0) -> float:
    """Seuil lisse: 0 si x < threshold, tend vers 1 sinon.
    
    Remplace: if x > threshold: return 1 else return 0
    """
    return sigmoid(x, center=threshold, steepness=steepness)


def smooth_clamp(x: float, min_val: float, max_val: float, steepness: float = 10.0) -> float:
    """Clamp lisse: reste dans [min_val, max_val] avec transitions douces.
    
    Remplace: np.clip(x, min_val, max_val)
    """
    # Sigmoid pour min
    above_min = sigmoid(x, center=min_val, steepness=steepness)
    # Sigmoid pour max (inversé)
    below_max = sigmoid(-x, center=-max_val, steepness=steepness)
    
    # Interpolation
    return min_val + (max_val - min_val) * above_min * below_max + x * (1 - above_min * below_max)


def inverse_distance(distance: float, scale: float = 1.0) -> float:
    """Urgence inversement proportionnelle à la distance.
    
    Remplace: if distance < threshold: return 0 else return some_value
    """
    return scale / (1.0 + distance)


def distance_urgency(distance: float, target_distance: float = 0.0, 
                     decay: float = 1.0) -> float:
    """Urgence basée sur l'éloignement d'une distance cible.
    
    Plus on est loin de target_distance, plus l'urgence est haute.
    Décroît exponentiellement vers 0 quand on s'approche.
    """
    diff = abs(distance - target_distance)
    return 1.0 - np.exp(-decay * diff)


# === FACTORIES POUR NEEDS ===

def create_urgency_fn(target: float = 0.0, 
                      near_threshold: float = 1.0,
                      far_threshold: float = 10.0) -> Callable[[float], float]:
    """Crée une fonction d'urgence lisse.
    
    - 0 si valeur proche de target (< near_threshold)
    - Augmente progressivement
    - Plafonne à 1 si valeur très loin (> far_threshold)
    """
    def urgency_fn(value: float) -> float:
        distance = abs(value - target)
        # Sigmoid pour la zone proche (pas d'urgence)
        not_near = smooth_threshold(distance, near_threshold, steepness=3.0)
        # Sigmoid pour le plafond
        not_far = 1.0 - smooth_threshold(distance, far_threshold, steepness=0.5)
        # Interpolation linéaire entre near et far
        linear_urgency = (distance - near_threshold) / (far_threshold - near_threshold)
        linear_urgency = max(0, min(1, linear_urgency))
        
        return not_near * (not_far * linear_urgency + (1 - not_far))
    
    return urgency_fn


def create_direction_fn(steepness: float = 1.0) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Crée une fonction de direction lisse vers une cible.
    
    La magnitude de la direction dépend de la distance (plus loin = plus fort)
    mais avec saturation douce.
    """
    def direction_fn(position: np.ndarray, target: np.ndarray) -> np.ndarray:
        diff = target - position
        distance = np.linalg.norm(diff)
        
        if distance < 1e-6:
            return np.zeros_like(diff)
        
        # Direction unitaire
        direction = diff / distance
        
        # Magnitude avec saturation douce (tanh)
        magnitude = tanh_scaled(distance * steepness, scale=1.0)
        
        return direction * magnitude
    
    return direction_fn


# === VECTORIZED VERSIONS ===

def sigmoid_vec(x: np.ndarray, center: float = 0.0, steepness: float = 1.0) -> np.ndarray:
    """Version vectorisée de sigmoid."""
    return 1.0 / (1.0 + np.exp(-steepness * (x - center)))


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax pour normaliser un vecteur en probabilités."""
    exp_x = np.exp((x - np.max(x)) / temperature)
    return exp_x / np.sum(exp_x)


def weighted_sum_smooth(values: np.ndarray, weights: np.ndarray, 
                        temperature: float = 1.0) -> np.ndarray:
    """Somme pondérée avec softmax pour lisser les poids."""
    normalized_weights = softmax(weights, temperature)
    return np.sum(values * normalized_weights[:, np.newaxis], axis=0)
