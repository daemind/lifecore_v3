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


# === GRIPPER ===

class GripperStatus(Enum):
    """Status du gripper."""
    OPEN = "open"
    CLOSED = "closed"
    HOLDING = "holding"
    RELEASING = "releasing"


class Gripper:
    """Préhenseur avec LifeCore.
    
    Le gripper a des needs:
    - grip_force_need: maintenir la force de grip appropriée
    - object_security_need: ne pas lâcher un objet en cours de transport
    """
    
    def __init__(self, name: str = "gripper", max_force: float = 50.0):
        self.name = name
        self.max_force = max_force
        
        # État
        self.aperture: float = 1.0  # 0=fermé, 1=ouvert
        self.force: float = 0.0
        self.status = GripperStatus.OPEN
        self.held_object: Optional['WorldObject'] = None
        self.pending_object: Optional['WorldObject'] = None  # Object to grasp when closed
        
        # Cibles
        self.target_aperture: float = 1.0
        self.target_force: float = 0.0
        
        # Need: maintenir force appropriée
        self.grip_force_need = Need(
            sub_matrix=np.array([0.0, 1.0], dtype=np.float32),  # Force dim
            extractor=lambda s: abs(s[1] - 0.5) if len(s) > 1 else 0.0,  # Distance à 0.5
            urgency_fn=lambda d: float(np.clip(d * 2, 0, 1)),
            priority=1.5,
            name=f"{self.name}_grip_force"
        )
        
        # LifeCore (2 dims: aperture, force)
        self.core = LifeCore(dims=2, needs=[self.grip_force_need])
    
    def open(self) -> None:
        """Ouvre le gripper."""
        self.target_aperture = 1.0
        self.target_force = 0.0
        self.pending_object = None
        
        if self.held_object:
            self.held_object.is_held = False
            self.held_object = None
            self.status = GripperStatus.RELEASING
        else:
            self.status = GripperStatus.OPEN
    
    def close(self, force: float = 20.0) -> None:
        """Ferme le gripper avec une force donnée."""
        self.target_aperture = 0.0
        self.target_force = min(force, self.max_force)
        self.status = GripperStatus.CLOSED
    
    def grasp(self, obj: 'WorldObject') -> bool:
        """Initie la saisie d'un objet.
        
        L'objet sera saisi quand le gripper sera fermé (dans step()).
        
        Returns:
            True si la saisie est initiée
        """
        if self.held_object:
            return False  # Déjà en train de tenir quelque chose
        
        # Initier la fermeture avec objet en attente
        self.pending_object = obj
        self.close()
        
        return True
    
    def release(self) -> Optional['WorldObject']:
        """Relâche l'objet tenu."""
        obj = self.held_object
        if obj:
            obj.is_held = False
            self.held_object = None
        
        self.pending_object = None
        self.open()
        return obj
    
    def step(self, dt: float = 0.01) -> None:
        """Avance le gripper d'un pas."""
        # Contrôle proportionnel de l'aperture
        aperture_error = self.target_aperture - self.aperture
        self.aperture += 5.0 * aperture_error * dt  # Plus rapide
        self.aperture = np.clip(self.aperture, 0, 1)
        
        # Force
        if self.aperture < 0.1:
            self.force = self.target_force
        else:
            self.force = 0.0
        
        # Finaliser la saisie quand le gripper est fermé
        if self.pending_object and self.aperture < 0.1:
            self.held_object = self.pending_object
            self.held_object.is_held = True
            self.pending_object = None
            self.status = GripperStatus.HOLDING
        
        # Mise à jour status
        if self.aperture > 0.9:
            self.status = GripperStatus.OPEN
        elif self.held_object:
            self.status = GripperStatus.HOLDING
        elif self.aperture < 0.1:
            self.status = GripperStatus.CLOSED
    
    @property
    def is_holding(self) -> bool:
        return self.held_object is not None
    
    @property
    def is_grasping(self) -> bool:
        """True si en train de saisir (pending_object set)."""
        return self.pending_object is not None
    
    def get_state(self) -> Dict:
        return {
            "aperture": self.aperture,
            "force": self.force,
            "status": self.status.value,
            "holding": self.held_object.name if self.held_object else None,
            "grasping": self.pending_object.name if self.pending_object else None
        }


# === WORLD OBJECTS ===

class WorldObject:
    """Objet dans le monde (balle, verre, etc.)
    
    Représentation simple pour simulation.
    """
    
    def __init__(
        self,
        name: str,
        position: np.ndarray,
        size: float = 0.05,
        is_container: bool = False
    ):
        self.name = name
        self.position = np.array(position, dtype=np.float32)
        self.size = size
        self.is_container = is_container  # Peut contenir d'autres objets
        self.is_held = False
        self.contained_objects: List['WorldObject'] = []
    
    def contains(self, other: 'WorldObject') -> bool:
        """Vérifie si cet objet contient un autre (pour containers)."""
        if not self.is_container:
            return False
        
        # Simple check: l'autre objet est au-dessus et proche
        delta = other.position - self.position
        horizontal_dist = np.sqrt(delta[0]**2 + delta[1]**2)
        vertical_dist = delta[2]
        
        return horizontal_dist < self.size and 0 < vertical_dist < self.size * 2
    
    def get_pickup_position(self) -> np.ndarray:
        """Position pour saisir l'objet (au-dessus)."""
        return self.position + np.array([0, 0, self.size + 0.02])
    
    def get_drop_position(self) -> np.ndarray:
        """Position pour déposer dans ce container (si applicable)."""
        if self.is_container:
            return self.position + np.array([0, 0, self.size * 1.5])
        return self.position


class Scene:
    """Scène avec objets et bras."""
    
    def __init__(self, arm: RoboticArm):
        self.arm = arm
        self.arm.gripper = Gripper()  # Ajouter gripper au bras
        self.objects: Dict[str, WorldObject] = {}
    
    def add_object(self, obj: WorldObject) -> None:
        self.objects[obj.name] = obj
    
    def get_object(self, name: str) -> Optional[WorldObject]:
        return self.objects.get(name)
    
    def step(self, dt: float = 0.01) -> None:
        """Avance la simulation."""
        self.arm.step(dt)
        self.arm.gripper.step(dt)
        
        # Mettre à jour position des objets tenus
        if self.arm.gripper.held_object:
            self.arm.gripper.held_object.position = self.arm.forward_kinematics()
        
        # Vérifier containment
        for obj in self.objects.values():
            if obj.is_container:
                for other in self.objects.values():
                    if other != obj and obj.contains(other):
                        if other not in obj.contained_objects:
                            obj.contained_objects.append(other)
    
    def get_state(self) -> Dict:
        return {
            "arm": self.arm.get_state(),
            "gripper": self.arm.gripper.get_state(),
            "objects": {name: {
                "position": obj.position.tolist(),
                "is_held": obj.is_held,
                "contained": [o.name for o in obj.contained_objects] if obj.is_container else None
            } for name, obj in self.objects.items()}
        }


# === TASK LEARNER ===

class TaskGoal:
    """Un objectif de tâche (pas une séquence hardcodée).
    
    Le TaskLearner doit découvrir comment atteindre ce goal.
    """
    
    def __init__(
        self,
        name: str,
        check_fn: Callable[['Scene'], bool],
        reward: float = 1.0
    ):
        self.name = name
        self.check_fn = check_fn  # Fonction qui vérifie si le goal est atteint
        self.reward = reward
        self.achieved = False
    
    def is_achieved(self, scene: Scene) -> bool:
        result = self.check_fn(scene)
        if result:
            self.achieved = True
        return result


class TaskLearner:
    """Apprend à accomplir des tâches par exploration.
    
    PAS de séquences hardcodées - le système découvre
    les actions nécessaires via:
    - Exploration des actions possibles
    - Mémorisation des bonnes expériences
    - Réutilisation des patterns qui fonctionnent
    
    Actions disponibles:
    - move_to(position): déplacer le bras
    - open(): ouvrir le gripper
    - close(): fermer le gripper
    - grasp(object): saisir un objet
    - goto_object(object): aller vers un objet
    """
    
    def __init__(self, scene: Scene):
        self.scene = scene
        self.goals: List[TaskGoal] = []
        self.action_history: List[Dict] = []
        self.successful_sequences: List[List[Dict]] = []
        
        # LifeCore pour l'apprentissage des tâches
        # Dims: [gripper_holding, object_distance, goal_progress]
        self.core = LifeCore(dims=3)
        
        # Need: atteindre le goal
        self.goal_need = Need(
            sub_matrix=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            extractor=lambda s: 1.0 - s[2] if len(s) > 2 else 1.0,  # Distance au goal
            urgency_fn=lambda d: float(np.clip(d, 0, 1)),
            priority=3.0,
            name="goal_achievement"
        )
        self.core.needs.append(self.goal_need)
    
    def add_goal(self, goal: TaskGoal) -> None:
        self.goals.append(goal)
    
    def get_possible_actions(self) -> List[Dict]:
        """Retourne les actions possibles dans l'état actuel."""
        actions = []
        arm = self.scene.arm
        gripper = arm.gripper
        
        # Actions sur le gripper
        if gripper.status == GripperStatus.OPEN:
            actions.append({"type": "close"})
            
            # Si proche d'un objet, peut le saisir
            ee_pos = arm.forward_kinematics()
            for name, obj in self.scene.objects.items():
                if not obj.is_held and np.linalg.norm(ee_pos - obj.position) < 0.1:
                    actions.append({"type": "grasp", "object": name})
        
        if gripper.status in (GripperStatus.CLOSED, GripperStatus.HOLDING):
            actions.append({"type": "open"})
        
        # Mouvements vers objets
        for name, obj in self.scene.objects.items():
            if not obj.is_held:
                actions.append({"type": "goto", "object": name, "offset": "above"})
                actions.append({"type": "goto", "object": name, "offset": "at"})
                
                if obj.is_container and gripper.is_holding:
                    actions.append({"type": "drop_into", "container": name})
        
        return actions
    
    def execute_action(self, action: Dict) -> bool:
        """Exécute une action et retourne True si réussie."""
        arm = self.scene.arm
        gripper = arm.gripper
        
        action_type = action["type"]
        
        if action_type == "open":
            gripper.open()
            return True
        
        elif action_type == "close":
            gripper.close()
            return True
        
        elif action_type == "grasp":
            obj = self.scene.get_object(action["object"])
            if obj:
                return gripper.grasp(obj)
            return False
        
        elif action_type == "goto":
            obj = self.scene.get_object(action["object"])
            if obj:
                if action.get("offset") == "above":
                    target = obj.get_pickup_position()
                else:
                    target = obj.position.copy()
                return arm.move_to(target)
            return False
        
        elif action_type == "drop_into":
            container = self.scene.get_object(action["container"])
            if container and container.is_container:
                target = container.get_drop_position()
                if arm.move_to(target):
                    # Wait to arrive then release
                    return True
            return False
        
        return False
    
    def evaluate_state(self) -> np.ndarray:
        """Évalue l'état actuel pour le LifeCore."""
        arm = self.scene.arm
        gripper = arm.gripper
        
        # Dim 0: est-ce que le gripper tient quelque chose?
        holding = 1.0 if gripper.is_holding else 0.0
        
        # Dim 1: distance au plus proche objet non-tenu
        ee_pos = arm.forward_kinematics()
        min_dist = 1.0
        for obj in self.scene.objects.values():
            if not obj.is_held:
                dist = np.linalg.norm(ee_pos - obj.position)
                min_dist = min(min_dist, dist)
        object_proximity = 1.0 - min(min_dist, 1.0)
        
        # Dim 2: progression vers le goal (combien de goals atteints)
        achieved = sum(1 for g in self.goals if g.is_achieved(self.scene))
        goal_progress = achieved / max(1, len(self.goals))
        
        return np.array([holding, object_proximity, goal_progress])
    
    def learn_step(self) -> Optional[Dict]:
        """Fait un pas d'apprentissage.
        
        Choisit une action basée sur l'intention du LifeCore
        et l'exécute.
        
        Returns:
            L'action exécutée ou None
        """
        # État actuel
        state = self.evaluate_state()
        
        # Vérifier si goal atteint
        all_goals_achieved = all(g.is_achieved(self.scene) for g in self.goals)
        if all_goals_achieved:
            # Succès! Sauvegarder la séquence
            if self.action_history:
                self.successful_sequences.append(self.action_history.copy())
            return None
        
        # Obtenir les actions possibles
        actions = self.get_possible_actions()
        if not actions:
            return None
        
        # Utiliser l'intention du LifeCore pour scorer les actions
        intention = self.core.get_recursive_intention(state)
        
        # Scorer les actions selon leur alignement avec l'intention
        scored_actions = []
        for action in actions:
            score = self._score_action(action, intention, state)
            scored_actions.append((score, action))
        
        # Choisir la meilleure (avec un peu d'exploration aléatoire)
        scored_actions.sort(key=lambda x: -x[0])
        
        if np.random.random() < 0.2:  # 20% exploration
            action = actions[np.random.randint(len(actions))]
        else:
            action = scored_actions[0][1]
        
        # Exécuter
        success = self.execute_action(action)
        
        # Logger
        self.action_history.append({
            "action": action,
            "state_before": state.tolist(),
            "success": success
        })
        
        # Apprendre de l'expérience
        new_state = self.evaluate_state()
        effect = new_state - state
        
        # Qualité = progression vers goal
        quality = 0.5 + (new_state[2] - state[2])  # Amélioration du goal progress
        
        self.core.add_experience(state, intention, effect, quality)
        
        return action
    
    def _score_action(self, action: Dict, intention: np.ndarray, state: np.ndarray) -> float:
        """Score une action selon l'état actuel et les besoins.
        
        Logic WITHOUT hardcoded sequence - based on algebraic state analysis:
        - Si pas de grip → priorité aux actions qui rapprochent d'un objet
        - Si grip → priorité aux actions qui rapprochent du container
        """
        score = 0.0
        action_type = action["type"]
        
        arm = self.scene.arm
        gripper = arm.gripper
        ee_pos = arm.forward_kinematics()
        
        # === PHASE 1: Pas de grip - chercher un objet ===
        if not gripper.is_holding:
            # Trouver l'objet non-container le plus proche
            target_obj = None
            min_dist = float('inf')
            
            for obj in self.scene.objects.values():
                if not obj.is_container and not obj.is_held:
                    dist = np.linalg.norm(ee_pos - obj.position)
                    if dist < min_dist:
                        min_dist = dist
                        target_obj = obj
            
            if target_obj:
                # Score basé sur la réduction de distance
                if action_type == "goto" and action.get("object") == target_obj.name:
                    if action.get("offset") == "above":
                        score += 3.0  # Priorité haute: aller au-dessus de l'objet
                    else:
                        score += 2.5
                
                # Si on est proche, grasp est la meilleure action
                if min_dist < 0.1:
                    if action_type == "grasp" and action.get("object") == target_obj.name:
                        score += 5.0  # Très haute priorité pour saisir
                    elif action_type == "close":
                        score += 2.0
        
        # === PHASE 2: Grip actif - aller vers container ===
        else:
            # Trouver le container
            target_container = None
            min_dist = float('inf')
            
            for obj in self.scene.objects.values():
                if obj.is_container:
                    dist = np.linalg.norm(ee_pos - obj.position)
                    if dist < min_dist:
                        min_dist = dist
                        target_container = obj
            
            if target_container:
                # Score pour aller vers container
                if action_type == "goto" and action.get("object") == target_container.name:
                    score += 4.0
                
                # Si on est au-dessus du container, drop
                if min_dist < 0.15:
                    if action_type == "drop_into":
                        score += 6.0  # Très haute priorité pour déposer
                    elif action_type == "open":
                        score += 5.0
        
        # Pénalité pour actions qui cassent le progrès
        if gripper.is_holding and action_type == "open" and state[2] < 0.5:
            # Ne pas ouvrir si on n'est pas encore au-dessus du container
            container_nearby = False
            for obj in self.scene.objects.values():
                if obj.is_container:
                    dist = np.linalg.norm(ee_pos - obj.position)
                    if dist < 0.15:
                        container_nearby = True
                        break
            if not container_nearby:
                score -= 3.0
        
        # Bonus d'intention du LifeCore (modulation adaptative)
        score += float(np.dot(intention, state)) * 0.5
        
        return score
    
    def run_episode(self, max_steps: int = 100, dt: float = 0.01) -> Dict:
        """Exécute un épisode complet d'apprentissage.
        
        Returns:
            Résultats de l'épisode
        """
        self.action_history = []
        
        # Reset goals
        for goal in self.goals:
            goal.achieved = False
        
        steps = 0
        actions_taken = 0
        waiting_for_movement = False
        
        for step in range(max_steps):
            # Simulation physique
            self.scene.step(dt)
            steps += 1
            
            # Vérifier succès AVANT de prendre une nouvelle action
            if all(g.is_achieved(self.scene) for g in self.goals):
                break
            
            # Si on attend la fin d'un mouvement, continuer à simuler
            arm = self.scene.arm
            if waiting_for_movement:
                gripper_stable = arm.gripper.aperture < 0.05 or arm.gripper.aperture > 0.95
                if arm.reached_target and gripper_stable:
                    waiting_for_movement = False
                else:
                    continue  # Continuer à simuler jusqu'à ce que le mouvement soit fini
            
            # Prendre une décision seulement si le bras est idle et gripper stabilisé
            if arm.status == ArmStatus.IDLE or arm.reached_target:
                action = self.learn_step()
                if action:
                    actions_taken += 1
                    # Après une action de mouvement, attendre
                    if action["type"] in ("goto", "drop_into"):
                        waiting_for_movement = True
        
        return {
            "steps": steps,
            "actions_taken": actions_taken,
            "goals_achieved": sum(1 for g in self.goals if g.achieved),
            "total_goals": len(self.goals),
            "success": all(g.achieved for g in self.goals),
            "action_sequence": [h["action"] for h in self.action_history]
        }


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
