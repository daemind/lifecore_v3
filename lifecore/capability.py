#!/usr/bin/env python3
"""
LifeCore V3 - Capability
========================

Capacités qui limitent ce qu'un LifeCore peut faire.
Contrairement aux Laws (externes), les Capabilities sont internes.

Exemples:
- Moteur: vitesse max, accélération max
- Capteur: portée, précision
- Batterie: capacité, taux de décharge

Principes:
- Chaque LifeCore a ses propres capacités
- Les capacités limitent les intentions de manière lisse (pas de clipping brutal)
- Les capacités peuvent s'adapter avec l'expérience
"""

import numpy as np
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from .activation import sigmoid, tanh_scaled, softplus


@dataclass
class Capability:
    """Capacité limitant l'intention sur certaines dimensions.
    
    Attributes:
        name: Nom de la capacité
        dims: Dimensions affectées
        max_value: Valeur maximale sur ces dimensions
        efficiency: Efficacité (0→1), réduit la sortie
        
    Example:
        >>> motor_cap = Capability(
        ...     name="motor_speed",
        ...     dims=[3, 4, 5],  # Velocity dimensions
        ...     max_value=15.0,
        ...     efficiency=1.0
        ... )
    """
    name: str
    dims: List[int]
    max_value: float = 10.0
    efficiency: float = 1.0
    
    # Paramètres de la fonction de limitation (tanh par défaut)
    saturation_steepness: float = 1.0
    
    def limit(self, intention: np.ndarray) -> np.ndarray:
        """Applique la limitation de capacité à une intention.
        
        Utilise tanh pour une saturation douce au lieu de clipping.
        """
        result = intention.copy()
        
        for dim in self.dims:
            if dim < len(result):
                # Saturation douce avec tanh
                normalized = result[dim] / self.max_value
                saturated = tanh_scaled(normalized * self.saturation_steepness)
                result[dim] = saturated * self.max_value * self.efficiency
        
        return result
    
    def can_achieve(self, intention: np.ndarray) -> float:
        """Retourne le ratio de ce qui peut être réalisé (0→1).
        
        1.0 = l'intention est totalement réalisable
        0.5 = la moitié peut être réalisée
        """
        if not self.dims:
            return 1.0
        
        total_requested = 0.0
        total_achievable = 0.0
        
        for dim in self.dims:
            if dim < len(intention):
                requested = abs(intention[dim])
                achievable = min(requested, self.max_value * self.efficiency)
                total_requested += requested
                total_achievable += achievable
        
        if total_requested < 1e-6:
            return 1.0
        
        return total_achievable / total_requested


@dataclass
class CapabilitySet:
    """Ensemble de capacités d'un LifeCore."""
    
    capabilities: Dict[str, Capability] = field(default_factory=dict)
    
    def add(self, capability: Capability) -> None:
        """Ajoute une capacité."""
        self.capabilities[capability.name] = capability
    
    def remove(self, name: str) -> None:
        """Retire une capacité."""
        self.capabilities.pop(name, None)
    
    def apply_all(self, intention: np.ndarray) -> np.ndarray:
        """Applique toutes les capacités à une intention."""
        result = intention.copy()
        for cap in self.capabilities.values():
            result = cap.limit(result)
        return result
    
    def get_limiting_factor(self, intention: np.ndarray) -> Optional[str]:
        """Retourne le nom de la capacité la plus limitante."""
        min_ratio = 1.0
        limiting = None
        
        for name, cap in self.capabilities.items():
            ratio = cap.can_achieve(intention)
            if ratio < min_ratio:
                min_ratio = ratio
                limiting = name
        
        return limiting
    
    def total_efficiency(self, dims: Optional[List[int]] = None) -> float:
        """Retourne l'efficacité totale sur certaines dimensions."""
        if not self.capabilities:
            return 1.0
        
        efficiency = 1.0
        for cap in self.capabilities.values():
            if dims is None or any(d in dims for d in cap.dims):
                efficiency *= cap.efficiency
        
        return efficiency


# === FACTORY FUNCTIONS ===

def create_motor_capability(dims: List[int], max_speed: float = 10.0,
                           max_acceleration: float = 5.0) -> List[Capability]:
    """Crée les capacités d'un moteur.
    
    Returns:
        Liste de capacités: [speed_limit, acceleration_limit]
    """
    return [
        Capability(
            name="speed_limit",
            dims=dims,
            max_value=max_speed,
            efficiency=1.0,
            saturation_steepness=2.0
        ),
        # Note: l'accélération serait appliquée sur le delta d'intention
    ]


def create_sensor_capability(range_value: float = 100.0,
                            precision: float = 0.95) -> Capability:
    """Crée la capacité d'un capteur."""
    return Capability(
        name="sensor_range",
        dims=[],  # Pas de limitation dimensionnelle directe
        max_value=range_value,
        efficiency=precision
    )


def create_battery_capability(capacity: float = 100.0,
                             discharge_rate: float = 0.1) -> Capability:
    """Crée la capacité d'une batterie."""
    return Capability(
        name="battery",
        dims=[],  # Affecte tout le système
        max_value=capacity,
        efficiency=1.0 - discharge_rate
    )
