#!/usr/bin/env python3
"""
LifeCore V3 - Laws
==================

Lois universelles qui s'appliquent à tous les LifeCore.
Contrairement aux ressources (quantités limitées), les lois sont
des contraintes que TOUS doivent respecter.

Exemples:
- Conservation de l'énergie
- Limite de vitesse
- Zones interdites
- Anti-collision
"""

import numpy as np
from typing import List, Optional, Callable, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .core import LifeCore


class Law(ABC):
    """Contrainte universelle sur les intentions/effets.
    
    Toutes les lois héritent de cette classe et implémentent
    la méthode constrain() qui modifie une intention pour
    qu'elle respecte la loi.
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def constrain(self, intention: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Applique la loi à une intention.
        
        Args:
            intention: Intention originale
            state: État courant
            
        Returns:
            Intention modifiée pour respecter la loi
        """
        pass
    
    def is_violated(self, intention: np.ndarray, state: np.ndarray) -> bool:
        """Vérifie si l'intention viole la loi."""
        constrained = self.constrain(intention, state)
        return not np.allclose(intention, constrained)


class SpeedLimit(Law):
    """Limite de vitesse sur certaines dimensions."""
    
    def __init__(self, max_speed: float, velocity_dims: List[int] = None):
        super().__init__(f"speed_limit_{max_speed}")
        self.max_speed = max_speed
        self.velocity_dims = velocity_dims or [2, 3, 4]  # Par défaut: dims 2,3,4
    
    def constrain(self, intention: np.ndarray, state: np.ndarray) -> np.ndarray:
        result = intention.copy()
        for dim in self.velocity_dims:
            if dim < len(result):
                result[dim] = np.clip(result[dim], -self.max_speed, self.max_speed)
        return result


class BoundaryLaw(Law):
    """Limite spatiale (boîte englobante)."""
    
    def __init__(self, min_bounds: np.ndarray, max_bounds: np.ndarray, 
                 position_dims: List[int] = None):
        super().__init__("boundary")
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds
        self.position_dims = position_dims or [0, 1, 2]
    
    def constrain(self, intention: np.ndarray, state: np.ndarray) -> np.ndarray:
        result = intention.copy()
        for i, dim in enumerate(self.position_dims):
            if dim >= len(state) or i >= len(self.min_bounds):
                continue
            # Si on va sortir des limites, bloquer
            new_pos = state[dim] + intention[dim]
            if new_pos < self.min_bounds[i]:
                result[dim] = max(0, self.min_bounds[i] - state[dim])
            elif new_pos > self.max_bounds[i]:
                result[dim] = min(0, self.max_bounds[i] - state[dim])
        return result


class NoFlyZone(Law):
    """Zones interdites."""
    
    def __init__(self, zones: List[dict]):
        """
        Args:
            zones: Liste de zones {center: [x,y,z], radius: float}
        """
        super().__init__("no_fly_zone")
        self.zones = zones
    
    def constrain(self, intention: np.ndarray, state: np.ndarray) -> np.ndarray:
        result = intention.copy()
        new_pos = state[:3] + intention[:3] if len(intention) >= 3 else state[:len(intention)]
        
        for zone in self.zones:
            center = np.array(zone['center'])
            radius = zone['radius']
            
            # Si on va entrer dans la zone
            dist = np.linalg.norm(new_pos[:len(center)] - center[:len(new_pos)])
            if dist < radius:
                # Direction opposée au centre de la zone
                away = new_pos[:len(center)] - center[:len(new_pos)]
                away_norm = np.linalg.norm(away)
                if away_norm > 1e-6:
                    away = away / away_norm
                    # Réduire l'intention dans la direction du centre
                    for i in range(min(len(result), len(away))):
                        if np.dot(intention[:len(away)], -away) > 0:
                            result[i] *= 0.1  # Réduire drastiquement
        
        return result


class CollisionAvoidance(Law):
    """Évitement de collision entre agents."""
    
    def __init__(self, min_distance: float = 5.0, other_positions: Callable = None):
        super().__init__("collision_avoidance")
        self.min_distance = min_distance
        self._get_other_positions = other_positions or (lambda: [])
    
    def set_position_getter(self, fn: Callable):
        self._get_other_positions = fn
    
    def constrain(self, intention: np.ndarray, state: np.ndarray) -> np.ndarray:
        result = intention.copy()
        my_pos = state[:3] if len(state) >= 3 else state
        new_pos = my_pos + intention[:len(my_pos)]
        
        for other_pos in self._get_other_positions():
            other = np.array(other_pos[:len(new_pos)])
            dist = np.linalg.norm(new_pos - other)
            
            if dist < self.min_distance:
                # Pousser dans la direction opposée
                away = new_pos - other
                away_norm = np.linalg.norm(away)
                if away_norm > 1e-6:
                    away = away / away_norm
                    # Ajouter une composante d'éloignement
                    push = away * (self.min_distance - dist) * 0.5
                    for i in range(min(len(result), len(push))):
                        result[i] += push[i]
        
        return result


class LawEnforcer:
    """Applique un ensemble de lois à toutes les intentions."""
    
    def __init__(self):
        self.laws: List[Law] = []
    
    def add_law(self, law: Law) -> None:
        self.laws.append(law)
    
    def remove_law(self, name: str) -> None:
        self.laws = [l for l in self.laws if l.name != name]
    
    def enforce(self, intention: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Applique toutes les lois séquentiellement."""
        result = intention.copy()
        for law in self.laws:
            result = law.constrain(result, state)
        return result
    
    def get_violations(self, intention: np.ndarray, state: np.ndarray) -> List[str]:
        """Retourne la liste des lois violées."""
        return [law.name for law in self.laws if law.is_violated(intention, state)]
