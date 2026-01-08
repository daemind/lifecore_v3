#!/usr/bin/env python3
"""
LifeCore V3 - Coherence
=======================

Contraintes de cohérence entre LifeCore frères.
Permet de coupler des sous-systèmes qui doivent agir ensemble.

Exemples:
- Roues gauche/droite: doivent tourner dans la même direction pour avancer
- Moteurs d'un drone: doivent se coordonner pour l'équilibre
- Muscles antagonistes: un se contracte, l'autre se relâche

Principes:
- Cohérence entre frères (même parent)
- Couplage via besoins partagés automatiques
- Pas d'interférence entre branches différentes de la hiérarchie
"""

import numpy as np
from typing import List, Optional, Callable, TYPE_CHECKING
from dataclasses import dataclass
from .activation import sigmoid, tanh_scaled

if TYPE_CHECKING:
    from .core import LifeCore


@dataclass
class CoherenceConstraint:
    """Contrainte de cohérence entre deux LifeCore.
    
    Attributes:
        name: Nom de la contrainte
        sibling_a: Premier frère
        sibling_b: Deuxième frère
        dims: Dimensions couplées
        mode: Type de couplage ('same', 'opposite', 'complementary')
        strength: Force du couplage (0→1)
    """
    name: str
    sibling_a: 'LifeCore'
    sibling_b: 'LifeCore'
    dims: List[int]
    mode: str = 'same'  # 'same', 'opposite', 'complementary'
    strength: float = 0.5
    
    def compute_correction(self, 
                          intention_a: np.ndarray, 
                          intention_b: np.ndarray) -> tuple:
        """Calcule les corrections pour aligner les intentions.
        
        Returns:
            (correction_a, correction_b): Corrections à appliquer
        """
        correction_a = np.zeros_like(intention_a)
        correction_b = np.zeros_like(intention_b)
        
        for dim in self.dims:
            if dim >= len(intention_a) or dim >= len(intention_b):
                continue
            
            val_a = intention_a[dim]
            val_b = intention_b[dim]
            
            if self.mode == 'same':
                # Les deux doivent avoir la même valeur
                target = (val_a + val_b) / 2
                correction_a[dim] = (target - val_a) * self.strength
                correction_b[dim] = (target - val_b) * self.strength
                
            elif self.mode == 'opposite':
                # Les deux doivent avoir des valeurs opposées
                target_a = (val_a - val_b) / 2
                target_b = -target_a
                correction_a[dim] = (target_a - val_a) * self.strength
                correction_b[dim] = (target_b - val_b) * self.strength
                
            elif self.mode == 'complementary':
                # La somme doit être constante (disons 0)
                excess = (val_a + val_b) / 2
                correction_a[dim] = -excess * self.strength
                correction_b[dim] = -excess * self.strength
        
        return correction_a, correction_b


class CoherenceManager:
    """Gestionnaire des contraintes de cohérence pour un parent."""
    
    def __init__(self, parent: 'LifeCore'):
        self.parent = parent
        self.constraints: List[CoherenceConstraint] = []
    
    def couple(self, child_a: 'LifeCore', child_b: 'LifeCore',
               dims: List[int], mode: str = 'same', 
               strength: float = 0.5, name: Optional[str] = None) -> CoherenceConstraint:
        """Crée une contrainte de cohérence entre deux enfants.
        
        Args:
            child_a: Premier enfant
            child_b: Deuxième enfant
            dims: Dimensions à coupler
            mode: 'same' (identiques), 'opposite' (opposés), 'complementary' (somme=0)
            strength: Force du couplage (0→1)
            name: Nom de la contrainte
            
        Returns:
            La contrainte créée
        """
        # Vérifier que les deux sont bien des enfants du parent
        if child_a not in self.parent.children or child_b not in self.parent.children:
            raise ValueError("Les deux enfants doivent appartenir au même parent")
        
        constraint = CoherenceConstraint(
            name=name or f"coherence_{len(self.constraints)}",
            sibling_a=child_a,
            sibling_b=child_b,
            dims=dims,
            mode=mode,
            strength=strength
        )
        self.constraints.append(constraint)
        return constraint
    
    def uncouple(self, child_a: 'LifeCore', child_b: 'LifeCore') -> None:
        """Retire toutes les contraintes entre deux enfants."""
        self.constraints = [
            c for c in self.constraints 
            if not ((c.sibling_a == child_a and c.sibling_b == child_b) or
                   (c.sibling_a == child_b and c.sibling_b == child_a))
        ]
    
    def apply_all(self, intentions: dict) -> dict:
        """Applique toutes les contraintes aux intentions des enfants.
        
        Args:
            intentions: {child_id: intention_array}
            
        Returns:
            {child_id: corrected_intention}
        """
        corrected = {k: v.copy() for k, v in intentions.items()}
        
        for constraint in self.constraints:
            id_a = id(constraint.sibling_a)
            id_b = id(constraint.sibling_b)
            
            if id_a not in corrected or id_b not in corrected:
                continue
            
            corr_a, corr_b = constraint.compute_correction(
                corrected[id_a], corrected[id_b]
            )
            
            corrected[id_a] = corrected[id_a] + corr_a
            corrected[id_b] = corrected[id_b] + corr_b
        
        return corrected
    
    def get_violations(self, intentions: dict, tolerance: float = 0.1) -> List[str]:
        """Retourne les contraintes violées."""
        violations = []
        
        for constraint in self.constraints:
            id_a = id(constraint.sibling_a)
            id_b = id(constraint.sibling_b)
            
            if id_a not in intentions or id_b not in intentions:
                continue
            
            intent_a = intentions[id_a]
            intent_b = intentions[id_b]
            
            for dim in constraint.dims:
                if dim >= len(intent_a) or dim >= len(intent_b):
                    continue
                
                if constraint.mode == 'same':
                    diff = abs(intent_a[dim] - intent_b[dim])
                    if diff > tolerance:
                        violations.append(f"{constraint.name}[dim={dim}]: diff={diff:.2f}")
                        
                elif constraint.mode == 'opposite':
                    diff = abs(intent_a[dim] + intent_b[dim])
                    if diff > tolerance:
                        violations.append(f"{constraint.name}[dim={dim}]: sum={diff:.2f}")
        
        return violations
