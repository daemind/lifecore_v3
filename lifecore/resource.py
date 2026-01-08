#!/usr/bin/env python3
"""
LifeCore V3 - Resources
=======================

Ressources partagées entre LifeCore.
Chaque ressource a une capacité limitée et plusieurs consommateurs.

Principes:
- Les LifeCore demandent des ressources via request()
- L'allocation se fait selon les priorités
- Si pas assez de ressources → le besoin ne peut pas être satisfait
"""

import numpy as np
from typing import List, Dict, Optional, Callable, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .core import LifeCore


@dataclass
class ResourceRequest:
    """Demande de ressource par un LifeCore."""
    consumer: 'LifeCore'
    amount: float
    priority: float = 1.0


class SharedResource:
    """Ressource partagée entre plusieurs LifeCore.
    
    Exemples: énergie, temps CPU, espace mémoire, attention.
    
    Attributes:
        name: Nom de la ressource
        capacity: Capacité totale disponible
        allocated: Dict des allocations par consumer
        
    Example:
        >>> energy = SharedResource("energy", capacity=100.0)
        >>> motor1 = LifeCore(dims=4)
        >>> motor2 = LifeCore(dims=4)
        >>> energy.register(motor1, priority=2.0)
        >>> energy.register(motor2, priority=1.0)
        >>> # motor1 a priorité double, obtient plus
        >>> energy.request(motor1, 60.0)  # Obtient 60
        >>> energy.request(motor2, 60.0)  # Obtient 40 (reste)
    """
    
    def __init__(self, name: str, capacity: float = 100.0):
        self.name = name
        self.capacity = capacity
        self.consumers: Dict[int, 'LifeCore'] = {}  # id -> LifeCore
        self.priorities: Dict[int, float] = {}  # id -> priority
        self.allocated: Dict[int, float] = {}  # id -> amount allocated
        self.requests: Dict[int, float] = {}  # id -> amount requested
    
    def register(self, consumer: 'LifeCore', priority: float = 1.0) -> None:
        """Enregistre un consommateur de cette ressource."""
        cid = id(consumer)
        self.consumers[cid] = consumer
        self.priorities[cid] = priority
        self.allocated[cid] = 0.0
        self.requests[cid] = 0.0
    
    def unregister(self, consumer: 'LifeCore') -> None:
        """Désenregistre un consommateur."""
        cid = id(consumer)
        self.consumers.pop(cid, None)
        self.priorities.pop(cid, None)
        self.allocated.pop(cid, None)
        self.requests.pop(cid, None)
    
    def request(self, consumer: 'LifeCore', amount: float) -> float:
        """Demande une allocation de ressource.
        
        Args:
            consumer: LifeCore qui demande
            amount: Quantité demandée
            
        Returns:
            Quantité réellement allouée (peut être < amount)
        """
        cid = id(consumer)
        if cid not in self.consumers:
            return 0.0
        
        self.requests[cid] = amount
        return self._allocate()
    
    def release(self, consumer: 'LifeCore', amount: Optional[float] = None) -> None:
        """Libère une ressource allouée.
        
        Args:
            consumer: LifeCore qui libère
            amount: Quantité à libérer (None = tout)
        """
        cid = id(consumer)
        if cid not in self.allocated:
            return
        
        if amount is None:
            self.allocated[cid] = 0.0
        else:
            self.allocated[cid] = max(0.0, self.allocated[cid] - amount)
    
    def _allocate(self) -> float:
        """Réalloue les ressources selon les demandes et priorités."""
        # Calculer le total demandé pondéré par priorité
        total_weighted_demand = sum(
            self.requests.get(cid, 0.0) * self.priorities.get(cid, 1.0)
            for cid in self.consumers
        )
        
        if total_weighted_demand <= 0:
            return 0.0
        
        # Allouer proportionnellement
        for cid in self.consumers:
            demand = self.requests.get(cid, 0.0)
            priority = self.priorities.get(cid, 1.0)
            
            if total_weighted_demand <= self.capacity:
                # Assez pour tout le monde
                self.allocated[cid] = demand
            else:
                # Allocation proportionnelle à la priorité
                share = (demand * priority) / total_weighted_demand
                self.allocated[cid] = share * self.capacity
        
        # Retourner l'allocation du dernier demandeur
        return self.allocated.get(id(list(self.consumers.values())[-1]), 0.0)
    
    def get_allocation(self, consumer: 'LifeCore') -> float:
        """Retourne l'allocation courante d'un consommateur."""
        return self.allocated.get(id(consumer), 0.0)
    
    def available(self) -> float:
        """Retourne la capacité restante."""
        return self.capacity - sum(self.allocated.values())
    
    def utilization(self) -> float:
        """Retourne le taux d'utilisation (0→1)."""
        return sum(self.allocated.values()) / self.capacity if self.capacity > 0 else 0.0
    
    def get_stats(self) -> dict:
        """Retourne les statistiques de la ressource."""
        return {
            "name": self.name,
            "capacity": self.capacity,
            "allocated": sum(self.allocated.values()),
            "available": self.available(),
            "utilization": self.utilization(),
            "num_consumers": len(self.consumers)
        }


class ResourceNeed:
    """Besoin qui dépend d'une ressource partagée.
    
    L'intention générée est proportionnelle à la ressource obtenue.
    Si pas de ressource → pas d'intention (le besoin ne peut pas s'exprimer).
    """
    
    def __init__(self, 
                 base_need,  # Need de base
                 resource: SharedResource,
                 resource_per_unit: float = 1.0):
        """
        Args:
            base_need: Besoin sous-jacent
            resource: Ressource nécessaire
            resource_per_unit: Ressource consommée par unité d'intention
        """
        self.base_need = base_need
        self.resource = resource
        self.resource_per_unit = resource_per_unit
        self.consumer: Optional['LifeCore'] = None
        
        # Copier les attributs du besoin de base
        self.name = f"{base_need.name}_resource"
        self.priority = base_need.priority
    
    def bind(self, consumer: 'LifeCore') -> None:
        """Lie ce besoin à un consommateur."""
        self.consumer = consumer
        self.resource.register(consumer, priority=self.priority)
    
    def compute_intention(self, state: np.ndarray) -> np.ndarray:
        """Calcule l'intention, limitée par la ressource disponible."""
        if self.consumer is None:
            return np.zeros_like(state)
        
        # Intention de base
        base_intention = self.base_need.compute_intention(state)
        intention_magnitude = np.linalg.norm(base_intention)
        
        if intention_magnitude < 1e-6:
            return base_intention
        
        # Ressource nécessaire
        resource_needed = intention_magnitude * self.resource_per_unit
        
        # Demander la ressource
        resource_obtained = self.resource.request(self.consumer, resource_needed)
        
        # Réduire l'intention proportionnellement
        if resource_needed > 0:
            scale = resource_obtained / resource_needed
            return base_intention * scale
        
        return base_intention
    
    def get_urgency(self, state: np.ndarray) -> float:
        """L'urgence du besoin de base."""
        return self.base_need.get_urgency(state)
