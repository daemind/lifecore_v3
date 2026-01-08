#!/usr/bin/env python3
"""
TORSEUR: Architecture de Responsabilité Hiérarchique
====================================================

State global (Torseur):
    [position_x, position_y, velocity_x, velocity_y]
         │            │           │            │
         └────────────┴───────────┴────────────┘
                      ▼
                  Controller
              (objectif: atteindre cible)
                      │
         ┌────────────┴────────────┐
         ▼                         ▼
    Motor_X                    Motor_Y
  (domaine: dim 0,2)        (domaine: dim 1,3)
  (poussée/frein X)         (poussée/frein Y)

Chaque niveau:
- Possède un DOMAINE (sous-partie du torseur)
- Génère des BESOINS pour ses enfants
- Les enfants satisfont ces besoins → émergence

Run:
    cd lifecore-v3-clean
    python examples/torseur_hierarchy.py
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from lifecore import LifeCore, Need


# === TORSEUR: State global ===
# [pos_x, pos_y, vel_x, vel_y]
DIM_POS_X = 0
DIM_POS_Y = 1
DIM_VEL_X = 2
DIM_VEL_Y = 3
DIMS = 4


class TorseurNode(LifeCore):
    """LifeCore avec domaine de responsabilité dans le torseur."""
    
    def __init__(self, dims: int, domain_dims: list = None, **kwargs):
        """
        Args:
            dims: Dimensions totales du torseur
            domain_dims: Indices des dimensions dont ce node est responsable
        """
        super().__init__(dims=dims, **kwargs)
        self.domain_dims = domain_dims or list(range(dims))
        self.child_commands = {}  # Commandes reçues des parents
    
    def set_command(self, command_name: str, value: np.ndarray):
        """Reçoit une commande d'un parent."""
        self.child_commands[command_name] = value.copy()
    
    def get_domain_state(self, full_state: np.ndarray) -> np.ndarray:
        """Extrait la partie du state correspondant au domaine."""
        return full_state[self.domain_dims]
    
    def project_intention_to_domain(self, intention: np.ndarray) -> np.ndarray:
        """Projette une intention sur le domaine de responsabilité."""
        result = np.zeros(self.dims, dtype=np.float32)
        for i, dim in enumerate(self.domain_dims):
            if i < len(intention):
                result[dim] = intention[i]
        return result


class MotorNeed(Need):
    """Besoin d'un moteur: atteindre la commande cible."""
    
    def __init__(self, motor_node: 'MotorNode', command_name: str, dim: int):
        self.motor_node = motor_node
        self.command_name = command_name
        self.dim = dim
        
        super().__init__(
            sub_matrix=np.zeros(DIMS, dtype=np.float32),
            extractor=lambda s: float(s[dim]),
            urgency_fn=lambda v: 0.5,
            priority=1.0,
            name=f"motor_{command_name}"
        )
    
    def compute_intention(self, state: np.ndarray) -> np.ndarray:
        """L'intention = direction vers la commande cible."""
        command = self.motor_node.child_commands.get(self.command_name)
        if command is None:
            return np.zeros(DIMS, dtype=np.float32)
        
        # La commande est la vitesse désirée
        current_vel = state[self.dim + 2]  # vel = pos + 2
        target_vel = command[self.dim] if self.dim < len(command) else 0.0
        
        # Effort = différence entre vitesse cible et actuelle
        effort = target_vel - current_vel
        
        # Appliquer sur la dimension vitesse
        intention = np.zeros(DIMS, dtype=np.float32)
        intention[self.dim + 2] = np.clip(effort, -1.0, 1.0)
        
        urgency = abs(effort)
        return intention * min(urgency, 1.0)


class MotorNode(TorseurNode):
    """Moteur: contrôle une dimension de vitesse."""
    
    def __init__(self, axis: int):
        """
        Args:
            axis: 0 pour X, 1 pour Y
        """
        self.axis = axis
        domain = [axis, axis + 2]  # pos et vel de cet axe
        
        super().__init__(dims=DIMS, domain_dims=domain)
        
        # Le moteur a un besoin: suivre la commande de vitesse
        self.needs = [MotorNeed(self, "velocity_command", axis)]


class ControllerNeed(Need):
    """Besoin du controller: atteindre la cible."""
    
    def __init__(self, target: np.ndarray):
        self.target = target
        
        super().__init__(
            sub_matrix=np.ones(DIMS, dtype=np.float32) / np.sqrt(DIMS),
            extractor=lambda s: float(np.linalg.norm(s[:2] - target[:2])),
            urgency_fn=lambda d: 0.0 if d < 0.5 else min(d / 5.0, 1.0),
            priority=1.0,
            name="reach_target"
        )
    
    def compute_intention(self, state: np.ndarray) -> np.ndarray:
        """Calcule la vitesse désirée vers la cible."""
        pos = state[:2]
        direction = self.target[:2] - pos
        distance = np.linalg.norm(direction)
        
        if distance < 0.5:
            # Arrêt proche de la cible
            return np.zeros(DIMS, dtype=np.float32)
        
        # Vitesse désirée = direction normalisée
        direction = direction / (distance + 1e-6)
        
        # Vitesse proportionnelle à la distance (freinage progressif)
        speed = min(distance / 2.0, 1.0)
        
        intention = np.zeros(DIMS, dtype=np.float32)
        intention[2:4] = direction * speed  # Commande de vitesse
        
        return intention


class ControllerNode(TorseurNode):
    """Controller: coordonne les moteurs pour atteindre l'objectif."""
    
    def __init__(self, target: np.ndarray):
        super().__init__(dims=DIMS, domain_dims=[0, 1, 2, 3])  # Tout le torseur
        
        self.target = target
        self.needs = [ControllerNeed(target)]
        
        # Créer les moteurs comme enfants
        self.motor_x = MotorNode(axis=0)
        self.motor_y = MotorNode(axis=1)
        self.children = [self.motor_x, self.motor_y]
    
    def get_intention(self, state: np.ndarray) -> np.ndarray:
        """Le controller calcule la commande et la transmet aux moteurs."""
        # 1. Calculer l'intention de haut niveau (vitesse désirée)
        controller_intention = super().get_intention(state)
        
        # 2. Transmettre la commande aux moteurs
        velocity_command = controller_intention[2:4]  # Extraire la commande de vitesse
        self.motor_x.set_command("velocity_command", velocity_command)
        self.motor_y.set_command("velocity_command", velocity_command)
        
        # 3. Les moteurs calculent leur effort
        motor_x_effort = self.motor_x.get_intention(state)
        motor_y_effort = self.motor_y.get_intention(state)
        
        # 4. Fusion des efforts moteurs → intention finale
        final_intention = motor_x_effort + motor_y_effort
        
        return final_intention


def run_torseur_demo():
    print("=" * 60)
    print("  TORSEUR: Architecture de Responsabilité")
    print("=" * 60)
    print()
    
    target = np.array([10.0, 10.0], dtype=np.float32)
    print(f"Objectif: {target}")
    print()
    
    # Créer le controller (qui contient les moteurs)
    controller = ControllerNode(target)
    
    print("Hiérarchie:")
    print(f"  Controller (domaine: tout)")
    print(f"    ├── Motor X (domaine: pos_x, vel_x)")
    print(f"    └── Motor Y (domaine: pos_y, vel_y)")
    print()
    
    # État initial: [pos_x, pos_y, vel_x, vel_y]
    state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    print("Simulation:")
    for step in range(50):
        # Obtenir l'intention du controller (qui coordonne les moteurs)
        intention = controller.get_intention(state)
        
        # Physique simple: vitesse += intention, position += vitesse
        state[2:4] += intention[2:4] * 0.3  # Accélération
        state[2:4] *= 0.9  # Friction
        state[:2] += state[2:4] * 0.5  # Déplacement
        
        if step % 10 == 9:
            pos = state[:2]
            vel = state[2:4]
            dist = np.linalg.norm(pos - target)
            print(f"  Step {step+1}: pos={pos}, vel={vel}, dist={dist:.2f}")
    
    # Résultat
    final_dist = np.linalg.norm(state[:2] - target)
    print()
    print(f"Position finale: {state[:2]}")
    print(f"Distance à la cible: {final_dist:.2f}")
    
    if final_dist < 1.0:
        print("✅ SUCCÈS: Cible atteinte!")
    else:
        print("⚠️ Pas encore convergé")


if __name__ == "__main__":
    run_torseur_demo()
