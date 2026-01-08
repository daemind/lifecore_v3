#!/usr/bin/env python3
"""
LifeCore V3 - Need
==================

Un Besoin (Need) génère une intention vers sa satisfaction.
Chaque besoin a une direction (sub_matrix), une fonction d'extraction,
et une fonction d'urgence.

Principes:
- NO HARDCODE: urgence calculée dynamiquement
- Purement algébrique: intention = direction × urgence × priorité
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Need:
    """Un besoin génère une intention vers sa satisfaction.
    
    Attributes:
        sub_matrix: Direction tensorielle de l'intention (vecteur unitaire préféré)
        extractor: Fonction qui extrait la valeur du besoin depuis l'état
        urgency_fn: Fonction qui calcule l'urgence (0→1) depuis la valeur extraite
        priority: Multiplicateur de priorité (défaut 1.0)
        name: Nom optionnel pour debug
    
    Example:
        >>> import numpy as np
        >>> # Besoin d'énergie: urgence haute quand énergie basse
        >>> energy_need = Need(
        ...     sub_matrix=np.array([1.0, 0.0, 0.0, 0.0]),
        ...     extractor=lambda s: s[0],  # Énergie = dimension 0
        ...     urgency_fn=lambda e: (1.0 - np.clip(e, 0, 1)) ** 2,
        ...     priority=2.0,
        ...     name="energy"
        ... )
        >>> state = np.array([0.2, 0.5, 0.0, 0.0])  # Énergie basse
        >>> intent = energy_need.compute_intention(state)
        >>> intent[0] > 0  # Intention vers augmenter dimension 0
        True
    """
    sub_matrix: np.ndarray
    extractor: Callable[[np.ndarray], float]
    urgency_fn: Callable[[float], float]
    priority: float = 1.0
    name: str = ""
    
    def compute_intention(self, state: np.ndarray) -> np.ndarray:
        """Calcule l'intention générée par ce besoin.
        
        Intention = direction × urgence × priorité
        
        Args:
            state: État courant du système
            
        Returns:
            Vecteur d'intention (même dimension que sub_matrix)
        """
        value = self.extractor(state)
        urgency = float(self.urgency_fn(value))
        return self.sub_matrix * urgency * self.priority
    
    def get_urgency(self, state: np.ndarray) -> float:
        """Retourne l'urgence courante de ce besoin."""
        value = self.extractor(state)
        return float(self.urgency_fn(value))


def create_homeostatic_need(
    target_dim: int,
    dims: int,
    target_value: float = 0.5,
    priority: float = 1.0,
    name: str = ""
) -> Need:
    """Crée un besoin homéostatique simple.
    
    Le besoin tire vers target_value sur la dimension target_dim.
    L'urgence augmente quand on s'éloigne de la cible.
    
    Args:
        target_dim: Dimension de l'état à réguler
        dims: Nombre total de dimensions
        target_value: Valeur cible (défaut 0.5)
        priority: Priorité du besoin
        name: Nom pour debug
        
    Returns:
        Need configuré pour l'homéostasie
    """
    sub_matrix = np.zeros(dims, dtype=np.float32)
    sub_matrix[target_dim] = 1.0
    
    def extractor(s: np.ndarray) -> float:
        return float(s[target_dim]) if target_dim < len(s) else 0.0
    
    def urgency_fn(value: float) -> float:
        # Distance à la cible, normalisée
        distance = abs(value - target_value)
        return float(np.clip(distance, 0, 1))
    
    return Need(
        sub_matrix=sub_matrix,
        extractor=extractor,
        urgency_fn=urgency_fn,
        priority=priority,
        name=name or f"homeostatic_dim{target_dim}"
    )
