#!/usr/bin/env python3
"""
LifeCore V3 - Strategy
======================

Couche de stratégie/planification pour LifeCore.
Chaque LifeCore peut avoir une stratégie pour décomposer ses goals.

Stratégies disponibles:
- Direct: aller droit au but
- Exploration: essayer différentes directions
- Backtrack: revenir en arrière si bloqué
- Decompose: diviser un goal en sous-goals
- AStar: planification de chemin

Principes:
- La stratégie génère des sous-goals à partir du goal principal
- Chaque LifeCore peut avoir sa propre stratégie
- Les stratégies peuvent être composées
"""

import numpy as np
from typing import List, Optional, Dict, Callable, Tuple, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class StrategyState:
    """État interne d'une stratégie."""
    visited: Set[Tuple] = field(default_factory=set)
    dead_ends: Set[Tuple] = field(default_factory=set)
    path: List[np.ndarray] = field(default_factory=list)
    attempts: Dict[Tuple, int] = field(default_factory=dict)
    current_subgoal_idx: int = 0
    subgoals: List[np.ndarray] = field(default_factory=list)


class Strategy(ABC):
    """Interface de base pour les stratégies."""
    
    def __init__(self, name: str):
        self.name = name
        self.state = StrategyState()
    
    @abstractmethod
    def get_next_subgoal(self, 
                        current_pos: np.ndarray, 
                        target: np.ndarray,
                        obstacles: Optional[List[np.ndarray]] = None) -> Optional[np.ndarray]:
        """Retourne le prochain sous-goal à atteindre.
        
        Args:
            current_pos: Position actuelle
            target: Objectif final
            obstacles: Liste des obstacles connus
            
        Returns:
            Sous-goal ou None si pas de chemin
        """
        pass
    
    def on_reached(self, pos: np.ndarray):
        """Appelé quand un sous-goal est atteint."""
        self.state.visited.add(tuple(pos.round(1)))
        self.state.path.append(pos.copy())
    
    def on_blocked(self, pos: np.ndarray):
        """Appelé quand on est bloqué."""
        self.state.dead_ends.add(tuple(pos.round(1)))
    
    def reset(self):
        """Réinitialise l'état de la stratégie."""
        self.state = StrategyState()


class DirectStrategy(Strategy):
    """Stratégie directe: aller droit vers la cible."""
    
    def __init__(self):
        super().__init__("direct")
    
    def get_next_subgoal(self, current_pos, target, obstacles=None):
        # Simplement retourner la cible
        return target


class ExplorationStrategy(Strategy):
    """Stratégie d'exploration: essayer différentes directions."""
    
    def __init__(self, step_size: float = 1.0, num_directions: int = 8):
        super().__init__("exploration")
        self.step_size = step_size
        self.num_directions = num_directions
    
    def get_next_subgoal(self, current_pos, target, obstacles=None):
        pos_tuple = tuple(current_pos.round(1))
        
        # Incrémenter le compteur d'essais
        self.state.attempts[pos_tuple] = self.state.attempts.get(pos_tuple, 0) + 1
        
        # Générer les directions possibles
        angles = np.linspace(0, 2*np.pi, self.num_directions, endpoint=False)
        
        # Direction vers la cible (prioritaire)
        to_target = target - current_pos
        target_angle = np.arctan2(to_target[1] if len(to_target) > 1 else 0, 
                                  to_target[0])
        
        # Trier les angles par proximité à la direction cible
        angles_sorted = sorted(angles, key=lambda a: abs(a - target_angle))
        
        # Essayer chaque direction
        for angle in angles_sorted:
            direction = np.array([np.cos(angle), np.sin(angle)])
            if len(current_pos) > 2:
                direction = np.concatenate([direction, np.zeros(len(current_pos) - 2)])
            
            candidate = current_pos + direction * self.step_size
            candidate_tuple = tuple(candidate.round(1))
            
            # Éviter les impasses connues
            if candidate_tuple in self.state.dead_ends:
                continue
            
            # Éviter de revenir sur ses pas (sauf si nécessaire)
            if candidate_tuple in self.state.visited and self.state.attempts[pos_tuple] < 3:
                continue
            
            return candidate
        
        # Tout est bloqué → backtrack
        return None


class BacktrackStrategy(Strategy):
    """Stratégie de backtracking: revenir en arrière si bloqué."""
    
    def __init__(self, fallback: Strategy = None):
        super().__init__("backtrack")
        self.fallback = fallback or ExplorationStrategy()
    
    def get_next_subgoal(self, current_pos, target, obstacles=None):
        # Essayer la stratégie de base
        subgoal = self.fallback.get_next_subgoal(current_pos, target, obstacles)
        
        if subgoal is not None:
            return subgoal
        
        # Backtrack: revenir au dernier point non-impasse
        if self.state.path:
            for i in range(len(self.state.path) - 1, -1, -1):
                prev_pos = self.state.path[i]
                prev_tuple = tuple(prev_pos.round(1))
                
                if prev_tuple not in self.state.dead_ends:
                    # Marquer la position actuelle comme impasse
                    self.on_blocked(current_pos)
                    return prev_pos
        
        return None


class DecomposeStrategy(Strategy):
    """Stratégie de décomposition: diviser le chemin en waypoints."""
    
    def __init__(self, num_waypoints: int = 5):
        super().__init__("decompose")
        self.num_waypoints = num_waypoints
    
    def get_next_subgoal(self, current_pos, target, obstacles=None):
        # Si pas de sous-goals ou terminé, recalculer
        if not self.state.subgoals or self.state.current_subgoal_idx >= len(self.state.subgoals):
            self._compute_waypoints(current_pos, target)
        
        if self.state.current_subgoal_idx < len(self.state.subgoals):
            subgoal = self.state.subgoals[self.state.current_subgoal_idx]
            
            # Vérifier si on a atteint ce sous-goal
            dist = np.linalg.norm(current_pos - subgoal)
            if dist < 1.0:
                self.on_reached(subgoal)
                self.state.current_subgoal_idx += 1
                if self.state.current_subgoal_idx < len(self.state.subgoals):
                    return self.state.subgoals[self.state.current_subgoal_idx]
                return target
            
            return subgoal
        
        return target
    
    def _compute_waypoints(self, start: np.ndarray, end: np.ndarray):
        """Calcule les waypoints entre start et end."""
        self.state.subgoals = []
        for i in range(1, self.num_waypoints + 1):
            t = i / (self.num_waypoints + 1)
            waypoint = start + t * (end - start)
            self.state.subgoals.append(waypoint)
        self.state.current_subgoal_idx = 0


class AStarStrategy(Strategy):
    """Stratégie A*: planification de chemin optimal."""
    
    def __init__(self, grid_size: float = 1.0):
        super().__init__("astar")
        self.grid_size = grid_size
        self.obstacles_set: Set[Tuple] = set()
    
    def set_obstacles(self, obstacles: List[np.ndarray]):
        """Définit les obstacles."""
        self.obstacles_set = set(tuple(o.round(0)) for o in obstacles)
    
    def get_next_subgoal(self, current_pos, target, obstacles=None):
        if obstacles:
            self.set_obstacles(obstacles)
        
        # Si path déjà calculé et valide
        if self.state.subgoals and self.state.current_subgoal_idx < len(self.state.subgoals):
            subgoal = self.state.subgoals[self.state.current_subgoal_idx]
            
            dist = np.linalg.norm(current_pos - subgoal)
            if dist < self.grid_size:
                self.on_reached(subgoal)
                self.state.current_subgoal_idx += 1
                if self.state.current_subgoal_idx < len(self.state.subgoals):
                    return self.state.subgoals[self.state.current_subgoal_idx]
                return target
            
            return subgoal
        
        # Calculer le chemin
        path = self._astar(current_pos, target)
        if path:
            self.state.subgoals = path
            self.state.current_subgoal_idx = 0
            return path[0] if path else target
        
        return target
    
    def _astar(self, start: np.ndarray, goal: np.ndarray) -> List[np.ndarray]:
        """Implémentation A* simplifiée."""
        from heapq import heappush, heappop
        
        start_tuple = tuple(start.round(0).astype(int))
        goal_tuple = tuple(goal.round(0).astype(int))
        
        # File de priorité: (f_score, counter, pos_tuple)
        counter = 0
        open_set = [(0, counter, start_tuple)]
        came_from = {}
        g_score = {start_tuple: 0}
        
        dims = len(start)
        
        while open_set:
            _, _, current = heappop(open_set)
            
            if current == goal_tuple:
                # Reconstruire le chemin
                path = []
                while current in came_from:
                    path.append(np.array(current, dtype=float))
                    current = came_from[current]
                return path[::-1]
            
            # Voisins (6 directions en 3D, 4 en 2D)
            for d in range(dims):
                for delta in [-1, 1]:
                    neighbor = list(current)
                    neighbor[d] += int(delta * self.grid_size)
                    neighbor_tuple = tuple(neighbor)
                    
                    if neighbor_tuple in self.obstacles_set:
                        continue
                    
                    tentative_g = g_score[current] + self.grid_size
                    
                    if neighbor_tuple not in g_score or tentative_g < g_score[neighbor_tuple]:
                        came_from[neighbor_tuple] = current
                        g_score[neighbor_tuple] = tentative_g
                        f_score = tentative_g + np.linalg.norm(
                            np.array(neighbor) - goal
                        )
                        counter += 1
                        heappush(open_set, (f_score, counter, neighbor_tuple))
        
        return []  # Pas de chemin trouvé


class CompositeStrategy(Strategy):
    """Combine plusieurs stratégies avec fallback."""
    
    def __init__(self, strategies: List[Strategy]):
        super().__init__("composite")
        self.strategies = strategies
    
    def get_next_subgoal(self, current_pos, target, obstacles=None):
        for strategy in self.strategies:
            subgoal = strategy.get_next_subgoal(current_pos, target, obstacles)
            if subgoal is not None:
                return subgoal
        return None
    
    def on_blocked(self, pos):
        for strategy in self.strategies:
            strategy.on_blocked(pos)
    
    def reset(self):
        super().reset()
        for strategy in self.strategies:
            strategy.reset()


# === FACTORY ===

def create_maze_strategy() -> Strategy:
    """Crée une stratégie pour résoudre un maze."""
    return CompositeStrategy([
        AStarStrategy(grid_size=1.0),  # Essayer A* d'abord
        BacktrackStrategy(             # Fallback sur exploration + backtrack
            fallback=ExplorationStrategy(step_size=1.0, num_directions=8)
        )
    ])


def create_navigation_strategy() -> Strategy:
    """Crée une stratégie pour navigation générale."""
    return CompositeStrategy([
        DecomposeStrategy(num_waypoints=5),
        ExplorationStrategy(step_size=2.0)
    ])
