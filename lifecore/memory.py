#!/usr/bin/env python3
"""
LifeCore V3 - TensorMemory
==========================

Mémoire unifiée pour stocker et réutiliser les expériences.
Décision NETTE: si une expérience similaire avec bon résultat existe → réutiliser.

Principes:
- NO HARDCODE: seuil de similarité adaptatif
- Purement algébrique: similarité par produit scalaire normalisé
- Sélection directe, PAS de fusion barycentrique
"""

import numpy as np
from typing import Optional, List, Tuple, NamedTuple
from dataclasses import dataclass


@dataclass
class Experience:
    """Une expérience stockée en mémoire."""
    state: np.ndarray
    intention: np.ndarray
    effect: np.ndarray
    quality: float


class TensorMemory:
    """Mémoire tensorielle pour stockage et réutilisation d'expériences.
    
    Contrairement à l'ancienne architecture, cette mémoire utilise
    la SÉLECTION directe plutôt que la fusion barycentrique.
    
    Attributes:
        experiences: Liste des expériences stockées
        max_size: Taille maximale (buffer circulaire)
        
    Example:
        >>> import numpy as np
        >>> memory = TensorMemory(max_size=100)
        >>> state = np.array([1.0, 0.0, 0.0])
        >>> intent = np.array([0.0, 1.0, 0.0])
        >>> memory.add(state, intent, effect=intent*0.1, quality=0.9)
        >>> # Query retourne l'intention si état similaire trouvé
        >>> result = memory.query(state)
        >>> np.allclose(result, intent)
        True
    """
    
    def __init__(self, max_size: int = 10000):
        self.experiences: List[Experience] = []
        self.max_size = max_size
        self._insert_idx = 0
    
    @property
    def size(self) -> int:
        return len(self.experiences)
    
    def add(self, state: np.ndarray, intention: np.ndarray, 
            effect: np.ndarray, quality: float) -> None:
        """Stocke une nouvelle expérience.
        
        Args:
            state: État avant l'action
            intention: Intention exécutée
            effect: Effet observé (changement d'état)
            quality: Qualité de l'expérience (0→1)
        """
        exp = Experience(
            state=state.copy(),
            intention=intention.copy(),
            effect=effect.copy(),
            quality=quality
        )
        
        if len(self.experiences) < self.max_size:
            self.experiences.append(exp)
        else:
            # Buffer circulaire
            idx = self._insert_idx % self.max_size
            self.experiences[idx] = exp
        
        self._insert_idx += 1
    
    def query(self, state: np.ndarray, 
              threshold: float = 0.7,
              min_quality: float = 0.5) -> Optional[np.ndarray]:
        """Trouve la meilleure intention pour un état similaire.
        
        Retourne l'intention de l'expérience la plus similaire
        avec une qualité suffisante. Retourne None si aucune
        expérience ne correspond aux critères.
        
        Args:
            state: État courant à matcher
            threshold: Seuil de similarité minimum (0→1)
            min_quality: Qualité minimum requise
            
        Returns:
            Intention à réutiliser, ou None si rien trouvé
        """
        if not self.experiences:
            return None
        
        best_intention = None
        best_score = 0.0
        
        state_norm = np.linalg.norm(state)
        if state_norm < 1e-9:
            return None
        
        # Vectorisation: convertir en matrices pour calcul batch
        # (Optimisation si beaucoup d'expériences)
        for exp in self.experiences:
            if exp.state.shape != state.shape:
                continue
            if exp.quality < min_quality:
                continue
            
            # Similarité cosinus
            exp_norm = np.linalg.norm(exp.state)
            if exp_norm < 1e-9:
                continue
                
            similarity = np.dot(state, exp.state) / (state_norm * exp_norm)
            
            if similarity < threshold:
                continue
            
            # Score = similarité × qualité
            score = similarity * exp.quality
            
            if score > best_score:
                best_score = score
                best_intention = exp.intention.copy()
        
        return best_intention
    
    def query_with_score(self, state: np.ndarray,
                         threshold: float = 0.7) -> Tuple[Optional[np.ndarray], float]:
        """Query avec retour du score de confiance."""
        if not self.experiences:
            return None, 0.0
        
        best_intention = None
        best_score = 0.0
        
        state_norm = np.linalg.norm(state)
        if state_norm < 1e-9:
            return None, 0.0
        
        for exp in self.experiences:
            if exp.state.shape != state.shape:
                continue
            
            exp_norm = np.linalg.norm(exp.state)
            if exp_norm < 1e-9:
                continue
                
            similarity = np.dot(state, exp.state) / (state_norm * exp_norm)
            
            if similarity < threshold:
                continue
            
            score = similarity * exp.quality
            
            if score > best_score:
                best_score = score
                best_intention = exp.intention.copy()
        
        return best_intention, best_score
    
    def predict_effect(self, state: np.ndarray, intention: np.ndarray) -> Optional[np.ndarray]:
        """Prédit l'effet d'une intention basé sur les expériences passées.
        
        Trouve l'expérience la plus similaire (état ET intention)
        et retourne son effet.
        
        Args:
            state: État courant
            intention: Intention envisagée
            
        Returns:
            Effet prédit, ou None si pas d'expérience similaire
        """
        if not self.experiences:
            return None
        
        best_effect = None
        best_score = 0.0
        
        state_norm = np.linalg.norm(state)
        intent_norm = np.linalg.norm(intention)
        
        if state_norm < 1e-9 or intent_norm < 1e-9:
            return None
        
        for exp in self.experiences:
            if exp.state.shape != state.shape or exp.intention.shape != intention.shape:
                continue
            
            exp_state_norm = np.linalg.norm(exp.state)
            exp_intent_norm = np.linalg.norm(exp.intention)
            
            if exp_state_norm < 1e-9 or exp_intent_norm < 1e-9:
                continue
            
            state_sim = np.dot(state, exp.state) / (state_norm * exp_state_norm)
            intent_sim = np.dot(intention, exp.intention) / (intent_norm * exp_intent_norm)
            
            # Score combiné
            score = state_sim * intent_sim
            
            if score > best_score:
                best_score = score
                best_effect = exp.effect.copy()
        
        return best_effect
    
    def clear(self) -> None:
        """Vide la mémoire."""
        self.experiences.clear()
        self._insert_idx = 0
    
    def get_stats(self) -> dict:
        """Retourne les statistiques de la mémoire."""
        if not self.experiences:
            return {"size": 0, "avg_quality": 0.0}
        
        qualities = [e.quality for e in self.experiences]
        return {
            "size": len(self.experiences),
            "max_size": self.max_size,
            "avg_quality": float(np.mean(qualities)),
            "min_quality": float(np.min(qualities)),
            "max_quality": float(np.max(qualities))
        }
