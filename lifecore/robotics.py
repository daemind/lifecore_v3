#!/usr/bin/env python3
"""
LifeCore V3 - Robotics Module
=============================

Contrôle de bras robotisés via hiérarchie LifeCore.
Les mouvements émergent de la négociation entre joints, pas d'un pipeline prédéfini.

Architecture:
    Arm (LifeCore root)
    ├── Joint_1 (LifeCore) → épaule rotation
    ├── Joint_2 (LifeCore) → épaule élévation
    ├── Joint_3 (LifeCore) → coude
    ├── Joint_4 (LifeCore) → poignet rotation
    ├── Joint_5 (LifeCore) → poignet flexion
    └── Gripper (LifeCore) → préhenseur

Chaque joint:
- Goal: Position angulaire cible
- Need: Minimiser erreur de position
- Law: Limites angulaires, vitesse max, couple max
- Memory: Trajectoires optimales passées
- Feedback: Position actuelle, force

Usage:
    from lifecore.robotics import RoboticArm, Joint
    
    arm = RoboticArm(name="UR5", dof=6)
    arm.move_to(target=[0.5, 0.3, 0.2])  # Cinématique inverse
    
    while not arm.reached_target():
        arm.step(dt=0.01)  # 100Hz control loop
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

# Import core
try:
    from .core import LifeCore
    from .need import Need
    from .goal import Goal
    from .law import Law, LawEnforcer
    from .event import EventBus, Event, EventType
except ImportError:
    from core import LifeCore
    from need import Need
    from goal import Goal
    from law import Law, LawEnforcer
    from event import EventBus, Event, EventType


# === CONSTANTS ===

PI = np.pi
TWO_PI = 2 * np.pi


# === ENUMS ===

class JointType(Enum):
    """Type de joint."""
    REVOLUTE = "revolute"    # Rotation autour d'un axe
    PRISMATIC = "prismatic"  # Translation le long d'un axe


class ArmStatus(Enum):
    """Status du bras."""
    IDLE = "idle"
    MOVING = "moving"
    HOLDING = "holding"
    ERROR = "error"
    EMERGENCY_STOP = "e_stop"


# === JOINT ===

@dataclass
class JointConfig:
    """Configuration d'un joint."""
    name: str
    joint_type: JointType = JointType.REVOLUTE
    
    # Limites
    min_angle: float = -PI          # rad (ou m pour prismatic)
    max_angle: float = PI           # rad
    max_velocity: float = PI        # rad/s
    max_acceleration: float = PI    # rad/s²
    max_torque: float = 100.0       # Nm
    
    # Position initiale
    home_position: float = 0.0
    
    # Paramètres DH (Denavit-Hartenberg)
    dh_a: float = 0.0      # longueur du lien
    dh_d: float = 0.0      # offset selon z
    dh_alpha: float = 0.0  # twist angle


class Joint:
    """Joint robotique avec LifeCore intégré.
    
    Le joint utilise un LifeCore pour:
    - Décider de la vitesse optimale
    - Respecter les contraintes (lois)
    - Apprendre des trajectoires passées
    
    Hiérarchie des besoins:
    - position_need: minimiser l'erreur de position (urgence haute = loin de cible)
    - smoothness_need: minimiser les à-coups (urgence haute = accélération élevée)
    """
    
    def __init__(self, config: JointConfig, event_bus: Optional[EventBus] = None):
        self.config = config
        self.name = config.name
        self.event_bus = event_bus or EventBus()
        
        # État cinématique
        self.position: float = config.home_position
        self.velocity: float = 0.0
        self.acceleration: float = 0.0
        self.torque: float = 0.0
        
        # Cible
        self.target_position: float = config.home_position
        self.target_velocity: float = 0.0
        
        # Need: minimiser l'erreur de position
        # Dim 2 = erreur normalisée, on veut la ramener vers 0
        self.position_need = Need(
            sub_matrix=np.array([0.0, 0.0, -1.0, 0.0], dtype=np.float32),  # Réduire erreur
            extractor=lambda s: abs(s[2]) if len(s) > 2 else 0.0,  # |erreur|
            urgency_fn=lambda err: float(np.clip(err * 2, 0, 1)),  # Urgence prop. à erreur
            priority=2.0,
            name=f"{self.name}_position"
        )
        
        # Need: minimiser les à-coups (smooth motion)
        # Dim 1 = vitesse normalisée, on veut éviter les changements brusques
        self.smoothness_need = Need(
            sub_matrix=np.array([0.0, -1.0, 0.0, 0.0], dtype=np.float32),  # Réduire vitesse excessive
            extractor=lambda s: abs(s[1]) if len(s) > 1 else 0.0,  # |velocity|
            urgency_fn=lambda v: float(np.clip(v - 0.5, 0, 1)),  # Urgence si velocity > 0.5
            priority=0.5,
            name=f"{self.name}_smoothness"
        )
        
        # LifeCore pour le contrôle (4 dims: position, velocity, error, effort)
        # Needs passed to constructor
        self.core = LifeCore(
            dims=4,
            needs=[self.position_need, self.smoothness_need]
        )
        
        # Stats
        self.total_movement: float = 0.0
        self.cycles: int = 0
    
    @property
    def error(self) -> float:
        """Erreur de position."""
        return self.target_position - self.position
    
    @property
    def at_target(self) -> bool:
        """Vérifie si le joint est à la cible."""
        return abs(self.error) < 0.001 and abs(self.velocity) < 0.01
    
    def set_target(self, position: float, velocity: float = 0.0) -> None:
        """Définit la cible du joint."""
        # Clamp to limits
        self.target_position = np.clip(
            position, 
            self.config.min_angle, 
            self.config.max_angle
        )
        self.target_velocity = np.clip(
            velocity,
            -self.config.max_velocity,
            self.config.max_velocity
        )
    
    def step(self, dt: float = 0.01) -> None:
        """Avance le joint d'un pas de temps.
        
        Utilise un contrôleur PD simple pour le mouvement.
        Le LifeCore modulera les gains selon l'expérience.
        """
        # Calculer le contrôle PD
        Kp = 10.0  # Gain proportionnel
        Kd = 2.0   # Gain dérivé
        
        # État AVANT action (pour LifeCore)
        state_before = np.array([
            self.position / PI,           # Position normalisée
            self.velocity / self.config.max_velocity,  # Vitesse normalisée
            self.error / PI,              # Erreur normalisée
            self.torque / self.config.max_torque  # Effort normalisé
        ])
        
        # Obtenir l'intention du LifeCore (module les gains via expérience)
        intention = self.core.get_recursive_intention(state_before)
        
        # Moduler les gains selon l'intention (apprentissage adaptatif)
        # L'intention en dim 0 module le Kp, dim 1 module le Kd
        if len(intention) >= 2:
            Kp_mod = 1.0 + 0.5 * np.tanh(intention[0])  # [0.5, 1.5]
            Kd_mod = 1.0 + 0.3 * np.tanh(intention[1])  # [0.7, 1.3]
            Kp = Kp * Kp_mod
            Kd = Kd * Kd_mod
        
        # Calculer l'accélération désirée (PD)
        desired_acc = Kp * self.error + Kd * (self.target_velocity - self.velocity)
        
        # Limiter l'accélération
        self.acceleration = np.clip(
            desired_acc,
            -self.config.max_acceleration,
            self.config.max_acceleration
        )
        
        # Intégrer: velocity
        new_velocity = self.velocity + self.acceleration * dt
        self.velocity = np.clip(
            new_velocity,
            -self.config.max_velocity,
            self.config.max_velocity
        )
        
        # Intégrer: position
        old_position = self.position
        self.position += self.velocity * dt
        
        # Limiter la position
        self.position = np.clip(
            self.position,
            self.config.min_angle,
            self.config.max_angle
        )
        
        # Collision avec les limites
        if self.position == self.config.min_angle or \
           self.position == self.config.max_angle:
            self.velocity = 0.0
        
        # Stats
        self.total_movement += abs(self.position - old_position)
        
        # État APRÈS action
        state_after = np.array([
            self.position / PI,
            self.velocity / self.config.max_velocity,
            self.error / PI,
            self.torque / self.config.max_torque
        ])
        
        # Effet = delta d'état
        effect = state_after - state_before
        
        # Qualité = réduction d'erreur (bonne si l'erreur diminue)
        error_reduction = abs(state_before[2]) - abs(state_after[2])
        quality = np.clip(0.5 + error_reduction * 2, 0.0, 1.0)
        
        # Stocker l'expérience dans le LifeCore
        self.core.add_experience(state_before, intention, effect, quality)
    
    def get_dh_transform(self) -> np.ndarray:
        """Retourne la matrice de transformation DH 4x4."""
        theta = self.position  # Pour joint revolute
        d = self.config.dh_d
        a = self.config.dh_a
        alpha = self.config.dh_alpha
        
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        
        return np.array([
            [ct, -st*ca,  st*sa, a*ct],
            [st,  ct*ca, -ct*sa, a*st],
            [0,   sa,     ca,    d   ],
            [0,   0,      0,     1   ]
        ])
    
    def get_state(self) -> Dict:
        """Retourne l'état du joint."""
        return {
            "name": self.name,
            "position": self.position,
            "velocity": self.velocity,
            "target": self.target_position,
            "error": self.error,
            "at_target": self.at_target
        }


# === ROBOTIC ARM ===

class RoboticArm:
    """Bras robotique avec hiérarchie de LifeCores.
    
    Le bras coordonne plusieurs joints pour atteindre une position/orientation.
    La cinématique inverse est résolue via optimisation.
    """
    
    def __init__(
        self,
        name: str = "arm",
        joint_configs: Optional[List[JointConfig]] = None,
        dof: int = 6
    ):
        self.name = name
        self.event_bus = EventBus()
        
        # Créer les joints
        if joint_configs:
            self.joints = [Joint(cfg, self.event_bus) for cfg in joint_configs]
        else:
            # Configuration par défaut (bras 6-DOF type UR5)
            self.joints = self._create_default_arm(dof)
        
        self.dof = len(self.joints)
        
        # LifeCore racine pour coordonner
        self.core = LifeCore(dims=self.dof * 2)  # Position + velocity pour chaque joint
        
        # Cibles cartésiennes
        self.target_position: Optional[np.ndarray] = None  # [x, y, z]
        self.target_orientation: Optional[np.ndarray] = None  # [rx, ry, rz]
        
        # Status
        self.status = ArmStatus.IDLE
        
        # Stats
        self.movements_completed: int = 0
    
    def _create_default_arm(self, dof: int) -> List[Joint]:
        """Crée un bras par défaut."""
        configs = []
        
        # Configuration type bras 6-DOF
        default_configs = [
            {"name": "base", "dh_d": 0.1, "dh_a": 0.0, "dh_alpha": PI/2},
            {"name": "shoulder", "dh_d": 0.0, "dh_a": 0.4, "dh_alpha": 0},
            {"name": "elbow", "dh_d": 0.0, "dh_a": 0.3, "dh_alpha": 0},
            {"name": "wrist_1", "dh_d": 0.1, "dh_a": 0.0, "dh_alpha": PI/2},
            {"name": "wrist_2", "dh_d": 0.1, "dh_a": 0.0, "dh_alpha": -PI/2},
            {"name": "wrist_3", "dh_d": 0.05, "dh_a": 0.0, "dh_alpha": 0},
        ]
        
        for i in range(min(dof, len(default_configs))):
            cfg = default_configs[i]
            configs.append(JointConfig(
                name=cfg["name"],
                dh_d=cfg["dh_d"],
                dh_a=cfg["dh_a"],
                dh_alpha=cfg["dh_alpha"],
                max_velocity=PI,
                max_acceleration=2*PI
            ))
        
        return [Joint(cfg, self.event_bus) for cfg in configs]
    
    @property
    def joint_positions(self) -> np.ndarray:
        """Retourne les positions de tous les joints."""
        return np.array([j.position for j in self.joints])
    
    @property
    def joint_velocities(self) -> np.ndarray:
        """Retourne les vitesses de tous les joints."""
        return np.array([j.velocity for j in self.joints])
    
    @property
    def reached_target(self) -> bool:
        """Vérifie si tous les joints sont à leur cible."""
        return all(j.at_target for j in self.joints)
    
    def forward_kinematics(self, joint_angles: Optional[np.ndarray] = None) -> np.ndarray:
        """Calcule la position cartésienne de l'effecteur.
        
        Returns:
            Position [x, y, z] de l'effecteur final
        """
        if joint_angles is not None:
            # Temporairement set les positions
            for j, angle in zip(self.joints, joint_angles):
                j.position = angle
        
        # Multiplier les transformations DH
        T = np.eye(4)
        for joint in self.joints:
            T = T @ joint.get_dh_transform()
        
        # Extraire la position
        return T[:3, 3]
    
    def jacobian(self, joint_angles: Optional[np.ndarray] = None) -> np.ndarray:
        """Calcule le jacobien numérique.
        
        Le jacobien relie les vitesses articulaires aux vitesses cartésiennes.
        """
        if joint_angles is None:
            joint_angles = self.joint_positions
        
        eps = 1e-6
        J = np.zeros((3, self.dof))
        
        pos_0 = self.forward_kinematics(joint_angles)
        
        for i in range(self.dof):
            angles_plus = joint_angles.copy()
            angles_plus[i] += eps
            pos_plus = self.forward_kinematics(angles_plus)
            J[:, i] = (pos_plus - pos_0) / eps
        
        # Restaurer les positions
        for j, angle in zip(self.joints, joint_angles):
            j.position = angle
        
        return J
    
    def inverse_kinematics(
        self,
        target_pos: np.ndarray,
        max_iterations: int = 100,
        tolerance: float = 0.001
    ) -> Optional[np.ndarray]:
        """Résout la cinématique inverse par méthode de Jacobien.
        
        Args:
            target_pos: Position cible [x, y, z]
            max_iterations: Nombre max d'itérations
            tolerance: Tolérance sur l'erreur
        
        Returns:
            Angles des joints ou None si pas de solution
        """
        current_angles = self.joint_positions.copy()
        
        for iteration in range(max_iterations):
            # Position actuelle
            current_pos = self.forward_kinematics(current_angles)
            
            # Erreur
            error = target_pos - current_pos
            error_norm = np.linalg.norm(error)
            
            if error_norm < tolerance:
                return current_angles
            
            # Jacobien
            J = self.jacobian(current_angles)
            
            # Pseudo-inverse avec damping (Levenberg-Marquardt)
            lambda_damping = 0.1
            JJT = J @ J.T + lambda_damping * np.eye(3)
            delta_angles = J.T @ np.linalg.solve(JJT, error)
            
            # Mettre à jour avec step size
            step = 0.5
            current_angles = current_angles + step * delta_angles
            
            # Clamp aux limites
            for i, joint in enumerate(self.joints):
                current_angles[i] = np.clip(
                    current_angles[i],
                    joint.config.min_angle,
                    joint.config.max_angle
                )
        
        # Pas de solution trouvée
        return None
    
    def move_to(self, target: np.ndarray) -> bool:
        """Planifie un mouvement vers une position cartésienne.
        
        Args:
            target: Position cible [x, y, z]
        
        Returns:
            True si une solution IK existe
        """
        self.target_position = np.array(target)
        
        # Résoudre IK
        solution = self.inverse_kinematics(self.target_position)
        
        if solution is None:
            self.status = ArmStatus.ERROR
            return False
        
        # Définir les cibles des joints
        for joint, angle in zip(self.joints, solution):
            joint.set_target(angle)
        
        self.status = ArmStatus.MOVING
        return True
    
    def move_joints(self, target_angles: np.ndarray) -> None:
        """Mouvement direct en espace articulaire.
        
        Args:
            target_angles: Angles cibles pour chaque joint
        """
        for joint, angle in zip(self.joints, target_angles):
            joint.set_target(angle)
        
        self.status = ArmStatus.MOVING
    
    def step(self, dt: float = 0.01) -> None:
        """Avance tous les joints d'un pas de temps.
        
        Cette méthode doit être appelée à haute fréquence (100Hz+)
        pour un contrôle fluide.
        """
        # Mettre à jour chaque joint
        for joint in self.joints:
            joint.step(dt)
        
        # Vérifier si le mouvement est terminé
        if self.status == ArmStatus.MOVING and self.reached_target:
            self.status = ArmStatus.IDLE
            self.movements_completed += 1
    
    def emergency_stop(self) -> None:
        """Arrêt d'urgence - stoppe tous les joints immédiatement."""
        for joint in self.joints:
            joint.velocity = 0.0
            joint.target_position = joint.position
        
        self.status = ArmStatus.EMERGENCY_STOP
        
        if self.event_bus:
            self.event_bus.emit(Event(
                source=self.name,
                event_type=EventType.ALERT,
                severity=1.0,
                data={"reason": "emergency_stop"}
            ))
    
    def home(self) -> None:
        """Retourne à la position home."""
        for joint in self.joints:
            joint.set_target(joint.config.home_position)
        
        self.status = ArmStatus.MOVING
    
    def get_state(self) -> Dict:
        """Retourne l'état complet du bras."""
        end_effector = self.forward_kinematics()
        
        return {
            "name": self.name,
            "status": self.status.value,
            "dof": self.dof,
            "end_effector": end_effector.tolist(),
            "joints": [j.get_state() for j in self.joints],
            "target": self.target_position.tolist() if self.target_position is not None else None,
            "reached": self.reached_target,
            "movements": self.movements_completed
        }


# === TRAJECTORY PLANNING ===

class TrajectoryPlanner:
    """Planificateur de trajectoires.
    
    Génère des trajectoires lisses entre points.
    """
    
    @staticmethod
    def linear_interpolation(
        start: np.ndarray,
        end: np.ndarray,
        num_points: int = 100
    ) -> np.ndarray:
        """Interpolation linéaire entre deux positions.
        
        Returns:
            Tableau (num_points, dim) des positions intermédiaires
        """
        t = np.linspace(0, 1, num_points)
        return np.outer(1 - t, start) + np.outer(t, end)
    
    @staticmethod
    def trapezoidal_profile(
        distance: float,
        max_velocity: float,
        max_acceleration: float,
        dt: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Profil trapézoïdal de vitesse.
        
        Returns:
            (positions, velocities, times)
        """
        # Temps d'accélération
        t_acc = max_velocity / max_acceleration
        d_acc = 0.5 * max_acceleration * t_acc ** 2
        
        if 2 * d_acc >= distance:
            # Profil triangulaire (pas assez de distance pour atteindre v_max)
            t_acc = np.sqrt(distance / max_acceleration)
            t_total = 2 * t_acc
            t_cruise = 0
        else:
            # Profil trapézoïdal complet
            d_cruise = distance - 2 * d_acc
            t_cruise = d_cruise / max_velocity
            t_total = 2 * t_acc + t_cruise
        
        times = np.arange(0, t_total, dt)
        positions = np.zeros_like(times)
        velocities = np.zeros_like(times)
        
        for i, t in enumerate(times):
            if t < t_acc:
                # Phase d'accélération
                positions[i] = 0.5 * max_acceleration * t ** 2
                velocities[i] = max_acceleration * t
            elif t < t_acc + t_cruise:
                # Phase de croisière
                positions[i] = d_acc + max_velocity * (t - t_acc)
                velocities[i] = max_velocity
            else:
                # Phase de décélération
                t_dec = t - t_acc - t_cruise
                positions[i] = d_acc + max_velocity * t_cruise + \
                               max_velocity * t_dec - 0.5 * max_acceleration * t_dec ** 2
                velocities[i] = max_velocity - max_acceleration * t_dec
        
        return positions, velocities, times
    
    @staticmethod
    def cubic_spline(
        waypoints: np.ndarray,
        num_points: int = 100
    ) -> np.ndarray:
        """Interpolation par spline cubique.
        
        Args:
            waypoints: Points de passage (n_points, dim)
            num_points: Nombre de points sur la trajectoire finale
        
        Returns:
            Trajectoire lisse (num_points, dim)
        """
        from scipy.interpolate import CubicSpline
        
        n = len(waypoints)
        t_waypoints = np.linspace(0, 1, n)
        t_interp = np.linspace(0, 1, num_points)
        
        trajectory = np.zeros((num_points, waypoints.shape[1]))
        
        for dim in range(waypoints.shape[1]):
            cs = CubicSpline(t_waypoints, waypoints[:, dim])
            trajectory[:, dim] = cs(t_interp)
        
        return trajectory


# === COLLISION AVOIDANCE ===

class CollisionChecker:
    """Vérificateur de collisions simple.
    
    Pour une implémentation réelle, utiliser des bibliothèques
    comme FCL ou des meshes de collision.
    """
    
    def __init__(self):
        self.obstacles: List[Dict] = []
    
    def add_sphere_obstacle(self, center: np.ndarray, radius: float) -> None:
        """Ajoute un obstacle sphérique."""
        self.obstacles.append({
            "type": "sphere",
            "center": np.array(center),
            "radius": radius
        })
    
    def add_box_obstacle(self, min_corner: np.ndarray, max_corner: np.ndarray) -> None:
        """Ajoute un obstacle en forme de boîte."""
        self.obstacles.append({
            "type": "box",
            "min": np.array(min_corner),
            "max": np.array(max_corner)
        })
    
    def check_point(self, point: np.ndarray) -> bool:
        """Vérifie si un point est en collision.
        
        Returns:
            True si collision
        """
        for obs in self.obstacles:
            if obs["type"] == "sphere":
                dist = np.linalg.norm(point - obs["center"])
                if dist < obs["radius"]:
                    return True
            elif obs["type"] == "box":
                if np.all(point >= obs["min"]) and np.all(point <= obs["max"]):
                    return True
        
        return False
    
    def check_trajectory(self, trajectory: np.ndarray) -> List[int]:
        """Vérifie une trajectoire complète.
        
        Returns:
            Liste des indices en collision
        """
        collisions = []
        for i, point in enumerate(trajectory):
            if self.check_point(point):
                collisions.append(i)
        return collisions


# === DEMO ===

if __name__ == "__main__":
    print("Robotics Module Demo")
    print("=" * 50)
    
    # Créer un bras 6-DOF
    arm = RoboticArm(name="UR5", dof=6)
    print(f"Bras: {arm.name} ({arm.dof} DOF)")
    print(f"Joints: {[j.name for j in arm.joints]}")
    print()
    
    # Position initiale
    print("Position initiale (FK):")
    pos = arm.forward_kinematics()
    print(f"  End effector: {pos}")
    print()
    
    # Définir une cible
    target = np.array([0.3, 0.3, 0.3])
    print(f"Cible: {target}")
    
    # Résoudre IK
    print("Résolution IK...")
    if arm.move_to(target):
        print("  Solution trouvée!")
        
        # Simuler le mouvement
        print("Simulation mouvement (1000 steps @ 100Hz)...")
        for step in range(1000):
            arm.step(dt=0.01)
            
            if step % 200 == 0:
                pos = arm.forward_kinematics()
                error = np.linalg.norm(target - pos)
                print(f"  Step {step}: pos={pos.round(3)}, error={error:.4f}")
            
            if arm.reached_target:
                print(f"  ✅ Cible atteinte en {step} steps!")
                break
        
        final_pos = arm.forward_kinematics()
        print(f"\nPosition finale: {final_pos.round(3)}")
        print(f"Erreur finale: {np.linalg.norm(target - final_pos):.6f}")
    else:
        print("  ❌ Pas de solution IK")
    
    print()
    print("État final des joints:")
    for j in arm.joints:
        state = j.get_state()
        print(f"  {state['name']}: pos={state['position']:.3f} rad, at_target={state['at_target']}")
    
    # Demo trajectoire
    print()
    print("Demo planification trajectoire:")
    planner = TrajectoryPlanner()
    
    start = np.array([0, 0, 0.5])
    end = np.array([0.4, 0.4, 0.2])
    
    trajectory = planner.linear_interpolation(start, end, num_points=50)
    print(f"  Trajectoire: {len(trajectory)} points de {start} à {end}")
    
    # Demo collision
    print()
    print("Demo collision avoidance:")
    checker = CollisionChecker()
    checker.add_sphere_obstacle([0.2, 0.2, 0.35], radius=0.1)
    
    collisions = checker.check_trajectory(trajectory)
    print(f"  Obstacle sphérique à [0.2, 0.2, 0.35] rayon 0.1")
    print(f"  Collisions détectées: {len(collisions)} points sur {len(trajectory)}")
