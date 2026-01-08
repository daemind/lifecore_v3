#!/usr/bin/env python3
"""
LifeCore V3 - Config
====================

Système de configuration YAML/JSON pour LifeCore.
Permet de définir entièrement un système sans code.

Usage:
    from lifecore.config import load_system
    
    system = load_system("drone_delivery.yaml")
    system.run()
"""

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field

from .core import LifeCore
from .need import Need, create_homeostatic_need
from .goal import Goal
from .resource import SharedResource
from .capability import Capability, CapabilitySet
from .law import Law, LawEnforcer, SpeedLimit, BoundaryLaw, NoFlyZone, CollisionAvoidance


# === SCHEMA ===

@dataclass
class SystemConfig:
    """Configuration complète d'un système LifeCore."""
    name: str
    dims: int
    dim_names: List[str] = field(default_factory=list)
    resources: List[Dict] = field(default_factory=list)
    laws: List[Dict] = field(default_factory=list)
    hierarchy: Dict = field(default_factory=dict)
    

# === LOADERS ===

def load_yaml(path: Union[str, Path]) -> Dict:
    """Charge un fichier YAML."""
    if not HAS_YAML:
        raise ImportError("PyYAML not installed. Use: pip install pyyaml")
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_json(path: Union[str, Path]) -> Dict:
    """Charge un fichier JSON."""
    with open(path, 'r') as f:
        return json.load(f)


def load_config(path: Union[str, Path]) -> Dict:
    """Charge un fichier de config (YAML ou JSON)."""
    path = Path(path)
    if path.suffix in ['.yaml', '.yml']:
        return load_yaml(path)
    elif path.suffix == '.json':
        return load_json(path)
    else:
        raise ValueError(f"Format non supporté: {path.suffix}")


# === BUILDERS ===

def build_law(config: Dict, dims: int) -> Optional[Law]:
    """Construit une Law depuis la config."""
    law_type = config.get('type', '')
    
    if law_type == 'speed_limit':
        return SpeedLimit(
            max_speed=config.get('max', 10.0),
            velocity_dims=config.get('dims', [])
        )
    
    elif law_type == 'boundary':
        return BoundaryLaw(
            min_bounds=np.array(config.get('min', [0]*dims)),
            max_bounds=np.array(config.get('max', [100]*dims)),
            position_dims=config.get('dims', list(range(dims)))
        )
    
    elif law_type == 'no_fly_zone':
        zones = config.get('zones', [])
        return NoFlyZone(zones=zones)
    
    elif law_type == 'collision':
        return CollisionAvoidance(
            min_distance=config.get('min_distance', 5.0)
        )
    
    elif law_type == 'traffic_light':
        # Feu de signalisation: bloque certaines dimensions quand rouge
        return TrafficLightLaw(
            position=np.array(config.get('position', [0, 0, 0])),
            dims=config.get('dims', []),
            cycle_time=config.get('cycle', 30.0)
        )
    
    elif law_type == 'wall':
        # Mur: bloque dans une direction
        return WallLaw(
            position=np.array(config.get('position', [0, 0, 0])),
            normal=np.array(config.get('normal', [1, 0, 0])),
            dims=config.get('dims', list(range(dims)))
        )
    
    return None


def build_capability(config: Dict) -> Capability:
    """Construit une Capability depuis la config."""
    return Capability(
        name=config.get('name', 'capability'),
        dims=config.get('dims', []),
        max_value=config.get('max', 10.0),
        efficiency=config.get('efficiency', 1.0)
    )


def build_need(config: Dict, dims: int) -> Need:
    """Construit un Need depuis la config."""
    need_type = config.get('type', 'homeostatic')
    
    if need_type == 'homeostatic':
        return create_homeostatic_need(
            target_dim=config.get('dim', 0),
            dims=dims,
            target_value=config.get('target', 0.0),
            priority=config.get('priority', 1.0),
            name=config.get('name', 'need')
        )
    
    # Autres types à ajouter
    return create_homeostatic_need(0, dims)


def build_node(config: Dict, dims: int, parent: Optional[LifeCore] = None,
               resources: Dict[str, SharedResource] = None,
               laws: LawEnforcer = None) -> LifeCore:
    """Construit un node LifeCore depuis la config."""
    
    # Créer le node
    domain_dims = config.get('domain', list(range(dims)))
    
    if parent:
        node = parent.spawn_child(domain_dims=domain_dims)
    else:
        node = LifeCore(dims=dims, domain_dims=domain_dims)
    
    # Ajouter les besoins
    for need_config in config.get('needs', []):
        node.needs.append(build_need(need_config, dims))
    
    # Ajouter les capacités
    for cap_config in config.get('capabilities', []):
        cap = build_capability(cap_config)
        # Stocker dans un attribut (à intégrer proprement plus tard)
        if not hasattr(node, 'capability_set'):
            node.capability_set = CapabilitySet()
        node.capability_set.add(cap)
    
    # Ajouter les ressources
    if resources:
        for res_name in config.get('resources', []):
            if res_name in resources:
                priority = config.get('resource_priority', {}).get(res_name, 1.0)
                node.add_resource(resources[res_name], priority=priority)
    
    # Ajouter les goals
    for goal_config in config.get('goals', []):
        target = np.zeros(dims)
        for i, v in enumerate(goal_config.get('target', [])):
            if i < dims:
                target[i] = v
        node.goals.push(Goal(
            target=target,
            name=goal_config.get('name', 'goal'),
            priority=goal_config.get('priority', 1.0),
            threshold=goal_config.get('threshold', 1.0)
        ))
    
    # Créer les enfants
    children_config = config.get('children', [])
    for child_config in children_config:
        count = child_config.get('count', 1)
        for i in range(count):
            child_name = child_config.get('name', 'child')
            child_cfg = child_config.copy()
            if count > 1:
                child_cfg['name'] = f"{child_name}_{i}"
            build_node(child_cfg, dims, parent=node, resources=resources, laws=laws)
    
    return node


# === NOUVELLES LOIS ===

class TrafficLightLaw(Law):
    """Feu de signalisation."""
    
    def __init__(self, position: np.ndarray, dims: List[int], cycle_time: float = 30.0):
        super().__init__("traffic_light")
        self.position = position
        self.dims = dims
        self.cycle_time = cycle_time
        self._time = 0.0
    
    def is_red(self) -> bool:
        """Le feu est-il rouge?"""
        return (self._time % self.cycle_time) < (self.cycle_time / 2)
    
    def tick(self, dt: float = 1.0):
        """Avance le temps."""
        self._time += dt
    
    def constrain(self, intention: np.ndarray, state: np.ndarray) -> np.ndarray:
        result = intention.copy()
        
        # Distance au feu
        dist = np.linalg.norm(state[:len(self.position)] - self.position)
        
        if dist < 5.0 and self.is_red():
            # Réduire l'intention sur les dims contrôlées
            for dim in self.dims:
                if dim < len(result):
                    result[dim] *= 0.1
        
        return result


class WallLaw(Law):
    """Mur infranchissable."""
    
    def __init__(self, position: np.ndarray, normal: np.ndarray, dims: List[int]):
        super().__init__("wall")
        self.position = position
        self.normal = normal / (np.linalg.norm(normal) + 1e-6)
        self.dims = dims
    
    def constrain(self, intention: np.ndarray, state: np.ndarray) -> np.ndarray:
        result = intention.copy()
        
        # Distance au mur (projetée sur la normale)
        to_wall = self.position[:len(state)] - state[:len(self.position)]
        dist_to_wall = np.dot(to_wall, self.normal[:len(to_wall)])
        
        if dist_to_wall < 2.0:
            # Projeter l'intention pour ne pas aller vers le mur
            intent_proj = np.dot(intention[:len(self.normal)], self.normal[:len(intention)])
            if intent_proj > 0:  # Allant vers le mur
                for i, dim in enumerate(self.dims):
                    if dim < len(result) and i < len(self.normal):
                        result[dim] -= intent_proj * self.normal[i]
        
        return result


class TrafficJamLaw(Law):
    """Embouteillage: ralentit dans certaines zones."""
    
    def __init__(self, zones: List[Dict]):
        super().__init__("traffic_jam")
        self.zones = zones  # [{'center': [...], 'radius': ..., 'slowdown': ...}]
    
    def constrain(self, intention: np.ndarray, state: np.ndarray) -> np.ndarray:
        result = intention.copy()
        
        for zone in self.zones:
            center = np.array(zone['center'])
            radius = zone.get('radius', 10.0)
            slowdown = zone.get('slowdown', 0.3)
            
            dist = np.linalg.norm(state[:len(center)] - center[:len(state)])
            if dist < radius:
                result *= slowdown
        
        return result


# === MAIN LOADER ===

def load_system(path: Union[str, Path]) -> Dict[str, Any]:
    """Charge un système complet depuis un fichier de config.
    
    Returns:
        {
            'root': LifeCore racine,
            'resources': {name: SharedResource},
            'laws': LawEnforcer,
            'config': Dict original
        }
    """
    config = load_config(path)
    
    dims = config.get('dims', 4)
    
    # Créer les ressources
    resources = {}
    for res_config in config.get('resources', []):
        name = res_config.get('name', 'resource')
        resources[name] = SharedResource(
            name=name,
            capacity=res_config.get('capacity', 100.0)
        )
    
    # Créer les lois
    laws = LawEnforcer()
    for law_config in config.get('laws', []):
        law = build_law(law_config, dims)
        if law:
            laws.add_law(law)
    
    # Créer la hiérarchie
    root = build_node(
        config.get('hierarchy', {'name': 'root'}),
        dims,
        resources=resources,
        laws=laws
    )
    
    return {
        'root': root,
        'resources': resources,
        'laws': laws,
        'config': config
    }
