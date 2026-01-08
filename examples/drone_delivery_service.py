#!/usr/bin/env python3
"""
DRONE DELIVERY SERVICE - LifeCore V3
====================================

Architecture complète d'un service de livraison par drone:

LOIS (partagées par tous)
├── SpeedLimit(max=15 m/s)
├── BoundaryLaw(zone de vol)
├── NoFlyZone(aéroport, hôpital)
└── CollisionAvoidance(min=10m)

RESSOURCES (partagées)
├── BatteryPool(capacité totale de la flotte)
├── AirspaceSlots(créneaux simultanés)
└── ChargingStations(postes de recharge)

HIÉRARCHIE
FleetController
├── Drone_1
│   ├── Navigation (domain: position)
│   ├── Propulsion (domain: velocity)
│   └── Cargo (domain: payload)
├── Drone_2
└── ...

Run:
    cd lifecore-v3-clean
    python examples/drone_delivery_service.py
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from lifecore import LifeCore, Goal, SharedResource, Need, create_homeostatic_need
from lifecore.law import SpeedLimit, BoundaryLaw, NoFlyZone, CollisionAvoidance, LawEnforcer


# === CONFIGURATION ===

DIMS = 7  # [x, y, z, vx, vy, vz, payload]
DIM_X, DIM_Y, DIM_Z = 0, 1, 2
DIM_VX, DIM_VY, DIM_VZ = 3, 4, 5
DIM_PAYLOAD = 6

# Zone de vol (km)
ZONE_MIN = np.array([0.0, 0.0, 10.0])   # Altitude min 10m
ZONE_MAX = np.array([10.0, 10.0, 100.0])  # Zone 10km x 10km, altitude max 100m

# Base (dépôt)
BASE_POSITION = np.array([5.0, 5.0, 10.0])

# Destinations de livraison
DELIVERY_POINTS = [
    np.array([2.0, 3.0, 10.0]),
    np.array([8.0, 1.0, 10.0]),
    np.array([7.0, 9.0, 10.0]),
    np.array([1.0, 8.0, 10.0]),
]


# === LOIS ===

def create_laws() -> LawEnforcer:
    """Crée l'ensemble des lois du système."""
    enforcer = LawEnforcer()
    
    # Limite de vitesse: 15 m/s
    enforcer.add_law(SpeedLimit(max_speed=15.0, velocity_dims=[DIM_VX, DIM_VY, DIM_VZ]))
    
    # Zone de vol autorisée
    enforcer.add_law(BoundaryLaw(
        min_bounds=ZONE_MIN,
        max_bounds=ZONE_MAX,
        position_dims=[DIM_X, DIM_Y, DIM_Z]
    ))
    
    # Zones interdites (aéroport simulé)
    enforcer.add_law(NoFlyZone(zones=[
        {'center': [0.0, 0.0, 20.0], 'radius': 1.0},  # Coin aéroport
    ]))
    
    return enforcer


# === NEEDS SPÉCIALISÉS ===

class DeliveryNeed(Need):
    """Besoin de livrer à destination."""
    
    def __init__(self, target: np.ndarray, priority: float = 1.0):
        self.target = target
        super().__init__(
            sub_matrix=np.zeros(DIMS, dtype=np.float32),
            extractor=lambda s: float(np.linalg.norm(s[:3] - target)),
            urgency_fn=lambda d: 0.0 if d < 1.0 else min(d / 5.0, 1.0),
            priority=priority,
            name="delivery"
        )
    
    def compute_intention(self, state: np.ndarray) -> np.ndarray:
        pos = state[:3]
        direction = self.target - pos
        dist = np.linalg.norm(direction)
        
        if dist < 1.0:
            return np.zeros(DIMS, dtype=np.float32)
        
        direction = direction / dist
        speed = min(dist * 0.5, 10.0)  # Vitesse proportionnelle, max 10
        
        intention = np.zeros(DIMS, dtype=np.float32)
        intention[DIM_VX:DIM_VZ+1] = direction * speed
        return intention


class ReturnToBaseNeed(Need):
    """Besoin de retourner à la base (seulement si pas de livraison active)."""
    
    def __init__(self, base: np.ndarray, drone: 'Drone' = None, priority: float = 0.5):
        self.base = base
        self.drone = drone
        super().__init__(
            sub_matrix=np.zeros(DIMS, dtype=np.float32),
            extractor=lambda s: float(np.linalg.norm(s[:3] - base)),
            urgency_fn=lambda d: 0.0 if d < 2.0 else 0.3,
            priority=priority,
            name="return_base"
        )
    
    def get_urgency(self, state: np.ndarray) -> float:
        # Pas d'urgence si on a une livraison en cours
        if self.drone and self.drone.current_delivery is not None:
            return 0.0
        dist = np.linalg.norm(state[:3] - self.base)
        return 0.0 if dist < 2.0 else 0.3
    
    def compute_intention(self, state: np.ndarray) -> np.ndarray:
        # Ne pas interférer si livraison en cours
        if self.drone and self.drone.current_delivery is not None:
            return np.zeros(DIMS, dtype=np.float32)
        
        pos = state[:3]
        direction = self.base - pos
        dist = np.linalg.norm(direction)
        
        if dist < 2.0:
            return np.zeros(DIMS, dtype=np.float32)
        
        direction = direction / dist
        intention = np.zeros(DIMS, dtype=np.float32)
        intention[DIM_VX:DIM_VZ+1] = direction * 5.0
        return intention


class BatteryNeed(Need):
    """Besoin de maintenir la batterie."""
    
    def __init__(self, drone: 'Drone'):
        self.drone = drone
        super().__init__(
            sub_matrix=np.zeros(DIMS, dtype=np.float32),
            extractor=lambda s: 0.0,
            urgency_fn=lambda v: 0.0,
            priority=2.0,
            name="battery"
        )
    
    def get_urgency(self, state: np.ndarray) -> float:
        # Urgence si batterie basse
        battery = self.drone.battery_level
        if battery < 20:
            return 1.0  # Critique
        elif battery < 50:
            return 0.5
        return 0.0
    
    def compute_intention(self, state: np.ndarray) -> np.ndarray:
        # Si batterie basse → retourner à la base charger
        if self.get_urgency(state) > 0:
            pos = state[:3]
            direction = BASE_POSITION - pos
            dist = np.linalg.norm(direction)
            if dist > 1.0:
                direction = direction / dist
                intention = np.zeros(DIMS, dtype=np.float32)
                intention[DIM_VX:DIM_VZ+1] = direction * 8.0
                return intention
        return np.zeros(DIMS, dtype=np.float32)


# === DRONE ===

class Drone(LifeCore):
    """Un drone de livraison."""
    
    def __init__(self, drone_id: int, fleet: 'FleetController'):
        super().__init__(dims=DIMS, parent=fleet)
        self.drone_id = drone_id
        self.fleet = fleet
        
        # État interne
        self.state = np.zeros(DIMS, dtype=np.float32)
        self.state[:3] = BASE_POSITION + np.random.randn(3) * 0.5
        
        # Batterie
        self.battery_level = 100.0
        
        # Mission courante
        self.current_delivery = None
        
        # Besoins
        self.needs = [
            BatteryNeed(self),
            ReturnToBaseNeed(BASE_POSITION, drone=self, priority=0.5)
        ]
        
        # Sous-composants
        self.navigation = self.spawn_child(domain_dims=[DIM_X, DIM_Y, DIM_Z])
        self.propulsion = self.spawn_child(domain_dims=[DIM_VX, DIM_VY, DIM_VZ])
    
    def assign_delivery(self, target: np.ndarray):
        """Assigne une livraison au drone."""
        self.current_delivery = target
        self.needs.insert(0, DeliveryNeed(target, priority=5.0))
        self.goals.push(Goal(
            target=np.concatenate([target, np.zeros(DIMS-3)]),
            name=f"deliver_to_{target[:2]}",
            priority=5.0,
            threshold=1.0
        ))
    
    def complete_delivery(self):
        """Marque la livraison comme terminée."""
        if self.current_delivery is not None:
            # Retirer le besoin de livraison
            self.needs = [n for n in self.needs if n.name != "delivery"]
            self.current_delivery = None
    
    def step(self, laws: LawEnforcer) -> dict:
        """Exécute un pas de simulation."""
        # Obtenir l'intention
        intention = self.get_intention(self.state)
        
        # Appliquer les lois
        legal_intention = laws.enforce(intention, self.state)
        
        # Consommer la batterie
        movement = np.linalg.norm(legal_intention[DIM_VX:DIM_VZ+1])
        self.battery_level -= movement * 0.1
        self.battery_level = max(0, self.battery_level)
        
        # Appliquer le mouvement (velocité = intention directement)
        self.state[DIM_VX:DIM_VZ+1] = legal_intention[DIM_VX:DIM_VZ+1]
        self.state[:3] += self.state[DIM_VX:DIM_VZ+1] * 0.2  # Position += velocity * dt
        
        # Vérifier si livraison terminée
        if self.current_delivery is not None:
            dist = np.linalg.norm(self.state[:3] - self.current_delivery)
            if dist < 1.0:
                self.complete_delivery()
        
        # Recharger à la base
        if np.linalg.norm(self.state[:3] - BASE_POSITION) < 2.0:
            self.battery_level = min(100, self.battery_level + 5.0)
        
        return {
            'position': self.state[:3].copy(),
            'battery': self.battery_level,
            'delivering': self.current_delivery is not None
        }


# === FLEET CONTROLLER ===

class FleetController(LifeCore):
    """Contrôleur de la flotte de drones."""
    
    def __init__(self, n_drones: int = 3):
        super().__init__(dims=DIMS)
        
        # Ressources partagées
        self.battery_pool = SharedResource("battery_pool", capacity=n_drones * 100)
        self.airspace = SharedResource("airspace", capacity=n_drones)
        
        # Créer les drones
        self.drones: list[Drone] = []
        for i in range(n_drones):
            drone = Drone(i, self)
            drone.add_resource(self.battery_pool, priority=5.0)
            drone.add_resource(self.airspace, priority=5.0)
            self.drones.append(drone)
            self.children.append(drone)
        
        # File de livraisons
        self.delivery_queue = list(DELIVERY_POINTS)
        self.completed_deliveries = 0
    
    def assign_pending_deliveries(self):
        """Assigne les livraisons en attente aux drones disponibles."""
        for drone in self.drones:
            if drone.current_delivery is None and self.delivery_queue:
                if drone.battery_level > 30:  # Assez de batterie
                    target = self.delivery_queue.pop(0)
                    drone.assign_delivery(target)
    
    def step(self, laws: LawEnforcer) -> dict:
        """Exécute un pas de simulation de la flotte."""
        # Assigner les livraisons
        self.assign_pending_deliveries()
        
        # Faire avancer tous les drones
        results = []
        for drone in self.drones:
            before_delivery = drone.current_delivery
            result = drone.step(laws)
            
            # Vérifier si livraison terminée
            if before_delivery is not None and drone.current_delivery is None:
                self.completed_deliveries += 1
            
            results.append(result)
        
        return {
            'drones': results,
            'completed': self.completed_deliveries,
            'pending': len(self.delivery_queue)
        }


# === SIMULATION ===

def run_simulation():
    print("=" * 70)
    print("  DRONE DELIVERY SERVICE - LifeCore V3")
    print("=" * 70)
    print()
    
    # Créer les lois
    laws = create_laws()
    print(f"Lois actives: {[l.name for l in laws.laws]}")
    print()
    
    # Créer la flotte
    fleet = FleetController(n_drones=3)
    print(f"Flotte créée: {len(fleet.drones)} drones")
    print(f"Livraisons en attente: {len(fleet.delivery_queue)}")
    print(f"Base: {BASE_POSITION}")
    print()
    
    # Simulation
    n_steps = 100
    print(f"Simulation: {n_steps} pas")
    print("-" * 50)
    
    for step in range(n_steps):
        result = fleet.step(laws)
        
        if step % 20 == 0 or step == n_steps - 1:
            print(f"\nStep {step}:")
            for i, d in enumerate(result['drones']):
                pos = d['position']
                status = "📦 Livrant" if d['delivering'] else "🔋 Recharge" if d['battery'] < 50 else "⏳ Attente"
                print(f"  Drone {i}: pos=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) bat={d['battery']:.0f}% {status}")
            print(f"  Livraisons: {result['completed']} terminées, {result['pending']} en attente")
    
    # Résultat final
    print()
    print("=" * 70)
    print("  RÉSULTAT")
    print("=" * 70)
    print(f"Livraisons complétées: {fleet.completed_deliveries}/{len(DELIVERY_POINTS)}")
    print(f"Ressource batterie utilisée: {fleet.battery_pool.utilization()*100:.0f}%")
    
    if fleet.completed_deliveries >= len(DELIVERY_POINTS) // 2:
        print("✅ Service opérationnel!")
    else:
        print("⚠️ Service sous-performant")


if __name__ == "__main__":
    run_simulation()
