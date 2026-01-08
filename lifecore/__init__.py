#!/usr/bin/env python3
"""
LifeCore V3 - Architecture Simplifiée
=====================================

Agent adaptatif avec:
- Goals → objectifs à atteindre
- Needs → besoins homéostatiques qui génèrent des intentions
- Resources → ressources partagées qui limitent les capacités
- Laws → contraintes universelles (externes)
- Capabilities → capacités internes (vitesse, force...)
- Coherence → couplage entre frères
- Mémoire → réutilisation directe
- Fractalité → enfants avec propres besoins, mémoire partagée

Usage:
    from lifecore import LifeCore, Need, Goal, SharedResource
    from lifecore import Capability, CapabilitySet
    from lifecore import CoherenceManager
    from lifecore.activation import sigmoid, smooth_threshold
"""

from .need import Need, create_homeostatic_need
from .memory import TensorMemory, Experience
from .core import LifeCore
from .goal import Goal, GoalStack
from .resource import SharedResource, ResourceNeed
from .law import Law, LawEnforcer, SpeedLimit, BoundaryLaw, NoFlyZone, CollisionAvoidance
from .capability import Capability, CapabilitySet, create_motor_capability
from .coherence import CoherenceConstraint, CoherenceManager

__all__ = [
    # Core
    'LifeCore',
    'Need',
    'TensorMemory',
    'Experience',
    'create_homeostatic_need',
    # Goals
    'Goal',
    'GoalStack',
    # Resources
    'SharedResource',
    'ResourceNeed',
    # Laws
    'Law',
    'LawEnforcer',
    'SpeedLimit',
    'BoundaryLaw',
    'NoFlyZone',
    'CollisionAvoidance',
    # Capabilities
    'Capability',
    'CapabilitySet',
    'create_motor_capability',
    # Coherence
    'CoherenceConstraint',
    'CoherenceManager',
]

__version__ = '3.1.0'
