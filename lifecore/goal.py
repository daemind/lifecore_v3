#!/usr/bin/env python3
"""
LifeCore V3 - Goal
==================

Un Goal est un objectif de haut niveau qui génère des besoins.
Contrairement à un Need (besoin homéostatique), un Goal représente
une aspiration externe.

Hiérarchie:
- Goal → génère des Needs → génèrent des Intentions

Principes:
- Un Goal peut être satisfait (terminé)
- Un Goal peut dépendre de ressources
- Un Goal peut créer plusieurs Needs pour être atteint
"""

import numpy as np
from typing import Optional, Callable, List
from dataclasses import dataclass
from .need import Need


@dataclass
class Goal:
    """Un objectif à atteindre.
    
    Attributes:
        target: Cible dans l'espace d'état
        name: Nom du goal
        priority: Priorité (0→∞)
        threshold: Distance à laquelle le goal est considéré atteint
        
    Example:
        >>> goal = Goal(
        ...     target=np.array([10.0, 10.0, 0.0, 0.0]),
        ...     name="reach_base",
        ...     priority=1.0,
        ...     threshold=1.0
        ... )
        >>> state = np.array([9.5, 9.5, 0.0, 0.0])
        >>> goal.is_reached(state)
        True
    """
    target: np.ndarray
    name: str = "goal"
    priority: float = 1.0
    threshold: float = 1.0
    
    def distance(self, state: np.ndarray) -> float:
        """Distance à la cible."""
        return float(np.linalg.norm(state - self.target))
    
    def is_reached(self, state: np.ndarray) -> bool:
        """Le goal est-il atteint?"""
        return self.distance(state) < self.threshold
    
    def urgency(self, state: np.ndarray) -> float:
        """Urgence du goal (0→1)."""
        if self.is_reached(state):
            return 0.0
        dist = self.distance(state)
        # Urgence inversement proportionnelle à la distance
        return float(np.clip(1.0 - dist / 10.0, 0.1, 1.0))
    
    def direction(self, state: np.ndarray) -> np.ndarray:
        """Direction vers la cible (normalisée)."""
        diff = self.target - state
        norm = np.linalg.norm(diff)
        if norm < 1e-6:
            return np.zeros_like(diff)
        return diff / norm
    
    def to_need(self) -> Need:
        """Convertit ce goal en un Need équivalent."""
        goal = self  # Capture pour closure
        
        class GoalNeed(Need):
            def compute_intention(self, state: np.ndarray) -> np.ndarray:
                if goal.is_reached(state):
                    return np.zeros_like(state)
                direction = goal.direction(state)
                urgency = goal.urgency(state)
                return direction * urgency * self.priority
        
        return GoalNeed(
            sub_matrix=np.ones(len(self.target)) / np.sqrt(len(self.target)),
            extractor=lambda s: goal.distance(s),
            urgency_fn=lambda d: goal.urgency(np.zeros(len(goal.target))),  # Placeholder
            priority=self.priority,
            name=self.name
        )


class GoalStack:
    """Pile de goals avec priorités.
    
    Permet de gérer plusieurs goals actifs simultanément.
    Le goal le plus prioritaire influence le plus l'intention.
    
    Example:
        >>> stack = GoalStack()
        >>> stack.push(Goal(target=np.array([10, 10, 0, 0]), name="explore"))
        >>> stack.push(Goal(target=np.array([0, 0, 0, 0]), name="return_home", priority=2.0))
        >>> # return_home a priorité → sera traité en premier
    """
    
    def __init__(self):
        self.goals: List[Goal] = []
    
    def push(self, goal: Goal) -> None:
        """Ajoute un goal à la pile."""
        self.goals.append(goal)
        # Trier par priorité décroissante
        self.goals.sort(key=lambda g: -g.priority)
    
    def pop(self, goal: Optional[Goal] = None) -> Optional[Goal]:
        """Retire un goal de la pile."""
        if goal is None:
            return self.goals.pop(0) if self.goals else None
        if goal in self.goals:
            self.goals.remove(goal)
            return goal
        return None
    
    def current(self) -> Optional[Goal]:
        """Retourne le goal le plus prioritaire."""
        return self.goals[0] if self.goals else None
    
    def update(self, state: np.ndarray) -> None:
        """Met à jour la pile: retire les goals atteints."""
        self.goals = [g for g in self.goals if not g.is_reached(state)]
    
    def compute_intention(self, state: np.ndarray) -> np.ndarray:
        """Calcule l'intention combinée de tous les goals."""
        if not self.goals:
            return np.zeros_like(state)
        
        total = np.zeros_like(state, dtype=np.float32)
        total_priority = sum(g.priority for g in self.goals)
        
        for goal in self.goals:
            if goal.is_reached(state):
                continue
            direction = goal.direction(state)
            urgency = goal.urgency(state)
            weight = goal.priority / total_priority
            total += direction * urgency * weight
        
        return total
    
    def __len__(self) -> int:
        return len(self.goals)
    
    def __iter__(self):
        return iter(self.goals)
