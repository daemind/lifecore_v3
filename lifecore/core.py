#!/usr/bin/env python3
"""
LifeCore V3 - Core
==================

Agent adaptatif minimal avec fractalité.
Chaque LifeCore a ses propres besoins, sa mémoire, et peut avoir des enfants.

Principes:
- Décision NETTE: mémoire → réutilisation directe OU besoins → intention
- Fractalité: enfants partagent la mémoire, contribuent à l'émergence
- Simplicité: ~150 lignes au lieu de 2400
"""

import numpy as np
from typing import List, Optional, Dict, Any, Callable
from .need import Need
from .memory import TensorMemory


class LifeCore:
    """Agent adaptatif avec mémoire et fractalité.
    
    Chaque LifeCore:
    - A des besoins qui génèrent des intentions
    - A une mémoire pour réutiliser les schémas passés
    - Peut avoir des enfants (sous-systèmes fractals)
    
    Le flux de décision est SIMPLE:
    1. Chercher en mémoire: "situation similaire + bon résultat?"
    2. Si OUI → réutiliser directement l'intention
    3. Si NON → calculer l'intention depuis les besoins
    
    Attributes:
        dims: Nombre de dimensions de l'espace d'état/intention
        needs: Liste des besoins de l'agent
        memory: Mémoire tensorielle
        children: Sous-systèmes fractals
        
    Example:
        >>> import numpy as np
        >>> from lifecore_v3 import LifeCore, Need
        >>> 
        >>> # Créer un agent simple
        >>> agent = LifeCore(dims=4)
        >>> 
        >>> # Lui apprendre quelque chose
        >>> state = np.array([1.0, 0.0, 0.0, 0.0])
        >>> good_intent = np.array([0.0, 1.0, 0.0, 0.0])
        >>> agent.add_experience(state, good_intent, good_intent*0.1, quality=0.9)
        >>>
        >>> # Il réutilise ce qu'il a appris
        >>> result = agent.get_intention(state)
        >>> np.allclose(result, good_intent)
        True
    """
    
    def __init__(self, 
                 dims: int,
                 needs: Optional[List[Need]] = None,
                 memory: Optional[TensorMemory] = None,
                 parent: Optional['LifeCore'] = None,
                 domain_dims: Optional[List[int]] = None):
        """Initialise un LifeCore.
        
        Args:
            dims: Dimensions de l'espace d'état
            needs: Liste des besoins (optionnel)
            memory: Mémoire partagée (optionnel, créée si None)
            parent: Parent fractal (optionnel)
            domain_dims: Indices des dimensions dont ce node est responsable
        """
        self.dims = dims
        self.needs = needs or []
        self.memory = memory or TensorMemory()
        self.parent = parent
        self.children: List['LifeCore'] = []
        
        # Torseur: domaine de responsabilité
        self.domain_dims = domain_dims or list(range(dims))
        
        # Commandes reçues du parent
        self.commands: Dict[str, np.ndarray] = {}
        
        # Récepteurs: fonctions qui mettent à jour le state interne
        self.receptors: List[Callable[[np.ndarray, np.ndarray], None]] = []
        
        # État interne (optionnel, peut être géré externement)
        self._state: Optional[np.ndarray] = None
        
        # Goals: objectifs à atteindre
        from .goal import GoalStack
        self.goals = GoalStack()
        
        # Resources: ressources partagées disponibles
        self.resources: Dict[str, Any] = {}
        
        # Strategy: planification pour atteindre les goals
        self.strategy = None  # Sera défini avec set_strategy()
        
        # Seuils adaptatifs
        self.similarity_threshold = 0.7
        self.min_quality = 0.5
    
    # === RESOURCES ===
    
    def add_resource(self, resource, priority: float = 1.0) -> None:
        """Ajoute une ressource partagée à ce LifeCore."""
        self.resources[resource.name] = resource
        resource.register(self, priority=priority)
    
    def get_resource(self, name: str):
        """Récupère une ressource par son nom."""
        return self.resources.get(name)
    
    # === STRATEGY ===
    
    def set_strategy(self, strategy) -> None:
        """Définit la stratégie de planification.
        
        Args:
            strategy: Instance de Strategy (DirectStrategy, AStarStrategy, etc.)
        """
        self.strategy = strategy
    
    def get_strategic_intention(self, state: np.ndarray, 
                                obstacles: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """Calcule l'intention en utilisant la stratégie.
        
        La stratégie décompose le goal en sous-goals,
        puis on calcule l'intention vers le sous-goal courant.
        
        Args:
            state: État courant
            obstacles: Obstacles connus (optionnel)
            
        Returns:
            Intention vers le prochain sous-goal
        """
        # Si pas de stratégie, utiliser l'intention normale
        if self.strategy is None:
            return self.get_intention(state)
        
        # Si pas de goal, pas d'intention
        current_goal = self.goals.current()
        if current_goal is None:
            return self.get_intention(state)
        
        # Position actuelle
        pos = state[:3] if len(state) >= 3 else state
        target = current_goal.target[:len(pos)]
        
        # Obtenir le prochain sous-goal de la stratégie
        subgoal = self.strategy.get_next_subgoal(pos, target, obstacles)
        
        if subgoal is None:
            # Stratégie bloquée
            self.strategy.on_blocked(pos)
            return np.zeros(self.dims, dtype=np.float32)
        
        # Calculer l'intention vers le sous-goal
        direction = subgoal - pos
        dist = np.linalg.norm(direction)
        
        if dist < 0.5:
            # Sous-goal atteint
            self.strategy.on_reached(pos)
            return np.zeros(self.dims, dtype=np.float32)
        
        direction = direction / dist
        speed = min(dist * 0.5, 5.0)  # Vitesse proportionnelle
        
        intention = np.zeros(self.dims, dtype=np.float32)
        for i in range(min(len(direction), self.dims)):
            intention[i] = direction[i] * speed
        
        return intention

    # === TORSEUR: Domaine et Commandes ===
    
    def set_command(self, name: str, value: np.ndarray) -> None:
        """Reçoit une commande du parent."""
        self.commands[name] = value.copy()
    
    def get_command(self, name: str) -> Optional[np.ndarray]:
        """Récupère une commande reçue."""
        return self.commands.get(name)
    
    def get_domain_state(self, full_state: np.ndarray) -> np.ndarray:
        """Extrait la partie du state correspondant au domaine."""
        return full_state[self.domain_dims]
    
    def project_to_domain(self, intention: np.ndarray) -> np.ndarray:
        """Projette une intention sur le domaine de responsabilité."""
        result = np.zeros(self.dims, dtype=np.float32)
        for i, dim in enumerate(self.domain_dims):
            if i < len(intention):
                result[dim] = intention[i]
        return result
    
    # === RÉCEPTEURS: Mise à jour du state ===
    
    def add_receptor(self, receptor_fn: Callable[[np.ndarray, np.ndarray], None]) -> None:
        """Ajoute un récepteur qui sera appelé lors de receive_effect."""
        self.receptors.append(receptor_fn)
    
    def receive_effect(self, effect: np.ndarray, state: Optional[np.ndarray] = None) -> np.ndarray:
        """Reçoit un effet de l'environnement et met à jour l'état.
        
        Args:
            effect: Effet observé (changement)
            state: État courant (optionnel, utilise _state si None)
            
        Returns:
            Nouvel état après application de l'effet
        """
        current_state = state if state is not None else self._state
        if current_state is None:
            current_state = np.zeros(self.dims, dtype=np.float32)
        
        # Appliquer l'effet
        new_state = current_state + effect
        
        # Appeler tous les récepteurs
        for receptor in self.receptors:
            receptor(new_state, effect)
        
        # Propager aux enfants (ils reçoivent l'effet sur leur domaine)
        for child in self.children:
            child.receive_effect(effect, new_state)
        
        self._state = new_state
        return new_state
    
    @property
    def state(self) -> Optional[np.ndarray]:
        """État interne courant."""
        return self._state
    
    @state.setter
    def state(self, value: np.ndarray) -> None:
        self._state = value.copy() if value is not None else None
    
    def get_intention(self, state: np.ndarray) -> np.ndarray:
        """Génère une intention pour l'état courant.
        
        Décision NETTE:
        1. Calculer l'intention depuis les besoins
        2. Si pas de besoin urgent → rester immobile
        3. Si besoin urgent ET mémoire trouve une solution → réutiliser
        4. Sinon → suivre les besoins
        
        Args:
            state: État courant (array de dimension self.dims)
            
        Returns:
            Intention (array de même dimension)
        """
        # 1. D'abord calculer l'intention des besoins
        if not self.needs:
            need_intention = np.zeros(self.dims, dtype=np.float32)
            total_urgency = 0.0
        else:
            need_intention = np.zeros(self.dims, dtype=np.float32)
            total_urgency = 0.0
            for need in self.needs:
                urgency = need.get_urgency(state)
                total_urgency += urgency
                need_intention += need.compute_intention(state)
        
        # 2. Si pas de besoin urgent → rester immobile
        if total_urgency < 0.1:
            return np.zeros(self.dims, dtype=np.float32)
        
        # 3. Chercher en mémoire une meilleure solution
        memory_intention = self.memory.query(
            state, 
            threshold=self.similarity_threshold,
            min_quality=self.min_quality
        )
        
        if memory_intention is not None:
            # Mémoire trouve quelque chose → réutiliser
            return memory_intention
        
        # 4. Sinon → suivre les besoins
        return need_intention
    
    # === RECURSIVE CONSTRAINT FEEDBACK ===
    
    def get_capacity(self, requested: np.ndarray) -> np.ndarray:
        """Retourne ce que ce node peut réellement faire.
        
        Les contraintes remontent: chaque enfant reporte ses limites,
        le parent agrège et reporte au sien.
        
        Args:
            requested: Intention/commande demandée par le parent
            
        Returns:
            Ce qui peut réellement être fait (limité par capacités et enfants)
        """
        result = requested.copy()
        
        # 1. Appliquer mes propres limites (besoins internes limitants)
        for need in self.needs:
            # Si un besoin crée une contrainte (ex: surchauffe → réduire)
            urgency = need.get_urgency(self._state if self._state is not None 
                                       else np.zeros(self.dims))
            if urgency > 0.5:  # Besoin urgent = contrainte
                # Réduire la capacité proportionnellement
                reduction = 1.0 - urgency * 0.5
                for dim in self.domain_dims:
                    if dim < len(result):
                        result[dim] *= reduction
        
        # 2. Si j'ai des enfants, leur demander ce qu'ils peuvent faire
        if self.children:
            # Distribuer la demande aux enfants selon leur domaine
            child_capacities = []
            for child in self.children:
                # Extraire la partie de la demande concernant cet enfant
                child_request = result.copy()
                child_capacity = child.get_capacity(child_request)
                child_capacities.append(child_capacity)
            
            # Agréger: prendre le minimum par dimension (le plus limitant gagne)
            if child_capacities:
                stacked = np.stack(child_capacities)
                # Pour chaque dimension, prendre le min en valeur absolue
                for dim in range(len(result)):
                    values = stacked[:, dim]
                    # Prendre celui avec la plus petite magnitude
                    min_idx = np.argmin(np.abs(values))
                    result[dim] = values[min_idx]
        
        return result
    
    def get_recursive_intention(self, state: np.ndarray, depth: int = 0, 
                                max_depth: int = 10) -> np.ndarray:
        """Calcule l'intention avec feedback récursif des enfants.
        
        Flux:
        1. Calculer intention idéale (depuis goal/besoins)
        2. Demander aux enfants ce qu'ils peuvent faire
        3. Ajuster l'intention selon les contraintes remontées
        4. Répéter jusqu'à convergence ou max_depth
        
        Args:
            state: État courant
            depth: Profondeur de récursion actuelle
            max_depth: Profondeur maximale
            
        Returns:
            Intention ajustée par les contraintes de tous les descendants
        """
        # Intention idéale
        ideal = self.get_intention(state)
        
        if not self.children or depth >= max_depth:
            return ideal
        
        # Demander aux enfants ce qu'ils peuvent faire
        capacity = self.get_capacity(ideal)
        
        # Ajuster: intention = min(ideal, capacity) par dimension
        # On utilise le signe de l'idéal avec la magnitude de la capacité
        adjusted = np.zeros_like(ideal)
        for dim in range(len(ideal)):
            if abs(ideal[dim]) > 1e-6:
                sign = np.sign(ideal[dim])
                magnitude = min(abs(ideal[dim]), abs(capacity[dim]))
                adjusted[dim] = sign * magnitude
            else:
                adjusted[dim] = capacity[dim]
        
        # Transmettre aux enfants comme commande
        for child in self.children:
            child.set_command("intention", adjusted)
        
        return adjusted
    
    def execute(self, state: np.ndarray) -> tuple:
        """Exécute une boucle complète: intention → effet.
        
        Returns:
            (intention, capacity, effect): Tuple des résultats
        """
        # 1. Calculer intention avec feedback
        intention = self.get_recursive_intention(state)
        
        # 2. Obtenir la capacité réelle
        capacity = self.get_capacity(intention)
        
        # 3. L'effet est ce qui est réellement fait
        effect = capacity.copy()
        
        # 4. Stocker l'expérience pour apprentissage
        quality = 1.0 - np.linalg.norm(intention - capacity) / (np.linalg.norm(intention) + 1e-6)
        quality = max(0, min(1, quality))
        
        if quality > 0.3:
            self.add_experience(state, intention, effect, quality)
        
        return intention, capacity, effect
    
    def add_experience(self, state: np.ndarray, intention: np.ndarray,
                       effect: np.ndarray, quality: float) -> None:
        """Enregistre une expérience dans la mémoire.
        
        Args:
            state: État avant l'action
            intention: Intention exécutée
            effect: Effet observé
            quality: Qualité de l'expérience (0→1)
        """
        self.memory.add(state, intention, effect, quality)
    
    def predict_effect(self, state: np.ndarray, intention: np.ndarray) -> Optional[np.ndarray]:
        """Prédit l'effet d'une intention basé sur l'expérience.
        
        Args:
            state: État courant
            intention: Intention envisagée
            
        Returns:
            Effet prédit, ou None si inconnu
        """
        return self.memory.predict_effect(state, intention)
    
    def spawn_child(self, child_needs: Optional[List[Need]] = None,
                    share_memory: bool = True,
                    domain_dims: Optional[List[int]] = None) -> 'LifeCore':
        """Crée un enfant fractal.
        
        L'enfant hérite de la mémoire du parent (si share_memory=True)
        mais a ses propres besoins et peut avoir son propre domaine.
        
        Args:
            child_needs: Besoins spécifiques de l'enfant
            share_memory: Si True, enfant partage la mémoire parent
            domain_dims: Domaine de responsabilité de l'enfant
            
        Returns:
            Nouveau LifeCore enfant
        """
        child = LifeCore(
            dims=self.dims,
            needs=child_needs or [],
            memory=self.memory if share_memory else TensorMemory(),
            parent=self,
            domain_dims=domain_dims
        )
        self.children.append(child)
        return child
    
    def get_collective_intention(self, state: np.ndarray) -> np.ndarray:
        """Calcule l'intention collective (parent + enfants).
        
        L'émergence fractale: chaque enfant contribue selon ses besoins.
        
        Args:
            state: État courant
            
        Returns:
            Intention émergente du groupe
        """
        # Intention du parent
        total = self.get_intention(state)
        
        # Contribution des enfants
        for child in self.children:
            total += child.get_intention(state)
        
        return total
    
    def get_needs_urgencies(self, state: np.ndarray) -> Dict[str, float]:
        """Retourne l'urgence de chaque besoin."""
        return {
            need.name or f"need_{i}": need.get_urgency(state)
            for i, need in enumerate(self.needs)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques du système."""
        return {
            "dims": self.dims,
            "num_needs": len(self.needs),
            "num_children": len(self.children),
            "memory": self.memory.get_stats(),
            "similarity_threshold": self.similarity_threshold,
            "min_quality": self.min_quality
        }
