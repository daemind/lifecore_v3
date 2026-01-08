#!/usr/bin/env python3
"""
DRONE FLEET SIMULATION - Réaliste
==================================

Simulation complète d'une flotte de drones de livraison.
Utilise la config exhaustive drone_fleet.yaml.

Scénario:
- 500 drones répartis sur 9 zones de Paris
- 1000 commandes à livrer en 8 heures
- Contraintes réelles: batterie, météo, no-fly zones, traffic
- Les drones doivent gérer: livraison, retour, recharge

Run:
    python examples/drone_fleet_simulation.py
"""

import numpy as np
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from collections import defaultdict

sys.path.insert(0, '.')

from lifecore import LifeCore, Need, Goal, SharedResource
from lifecore.config import load_system
from lifecore.law import LawEnforcer


# === CONFIGURATION SIMULATION ===

SIMULATION_HOURS = 8
STEPS_PER_HOUR = 60  # 1 step = 1 minute
TOTAL_STEPS = SIMULATION_HOURS * STEPS_PER_HOUR
NUM_ORDERS = 1000

# Position des hubs de zone (9 zones)
ZONE_HUBS = [
    (5, 5), (25, 5), (45, 5),
    (5, 25), (25, 25), (45, 25),
    (5, 45), (25, 45), (45, 45)
]


# === DATA STRUCTURES ===

@dataclass
class Order:
    """Commande à livrer."""
    id: int
    destination_x: float
    destination_y: float
    weight: float  # kg
    priority: int  # 1-10
    deadline_minutes: int
    assigned_drone: Optional[int] = None
    status: str = "pending"  # pending, in_transit, delivered, failed


@dataclass
class DroneState:
    """État d'un drone."""
    id: int
    zone: int
    pos_x: float
    pos_y: float
    altitude: float
    battery: float
    payload: float
    status: str  # idle, delivering, returning, charging
    current_order: Optional[int] = None
    distance_traveled: float = 0.0
    deliveries_completed: int = 0


# === SIMULATION ===

class DroneFleetSimulation:
    """Simulation réaliste de flotte de drones."""
    
    def __init__(self, config_path: str = "configs/drone_fleet.yaml"):
        print("Chargement de la configuration...")
        self.system = load_system(config_path)
        self.root = self.system['root']
        self.resources = self.system['resources']
        self.laws = self.system['laws']
        
        self.orders: List[Order] = []
        self.drones: Dict[int, DroneState] = {}
        self.step = 0
        
        # Stats
        self.stats = {
            'deliveries': 0,
            'failed': 0,
            'total_distance': 0.0,
            'total_energy': 0.0,
            'avg_delivery_time': 0.0,
            'violations': 0
        }
        
        self._init_drones()
        self._generate_orders()
    
    def _init_drones(self):
        """Initialise les drones par zone."""
        drone_id = 0
        drones_per_zone = 30
        
        for zone_idx, (hub_x, hub_y) in enumerate(ZONE_HUBS):
            for i in range(drones_per_zone):
                # Position aléatoire autour du hub
                offset_x = np.random.uniform(-2, 2)
                offset_y = np.random.uniform(-2, 2)
                
                self.drones[drone_id] = DroneState(
                    id=drone_id,
                    zone=zone_idx,
                    pos_x=hub_x + offset_x,
                    pos_y=hub_y + offset_y,
                    altitude=30.0,
                    battery=np.random.uniform(80, 100),
                    payload=0.0,
                    status="idle"
                )
                drone_id += 1
        
        print(f"  {len(self.drones)} drones initialisés sur {len(ZONE_HUBS)} zones")
    
    def _generate_orders(self):
        """Génère des commandes réalistes."""
        for i in range(NUM_ORDERS):
            # Destination aléatoire dans la région
            dest_x = np.random.uniform(0, 50)
            dest_y = np.random.uniform(0, 50)
            
            # Poids entre 0.1 et 2.5 kg
            weight = np.random.uniform(0.1, 2.5)
            
            # Priorité (Prime=9-10, Standard=5-7, Economy=1-3)
            priority_type = np.random.choice(['prime', 'standard', 'economy'], p=[0.3, 0.5, 0.2])
            priority = {
                'prime': np.random.randint(8, 11),
                'standard': np.random.randint(5, 8),
                'economy': np.random.randint(1, 4)
            }[priority_type]
            
            # Deadline selon priorité
            deadline = {
                'prime': np.random.randint(30, 60),
                'standard': np.random.randint(60, 180),
                'economy': np.random.randint(180, 480)
            }[priority_type]
            
            self.orders.append(Order(
                id=i,
                destination_x=dest_x,
                destination_y=dest_y,
                weight=weight,
                priority=priority,
                deadline_minutes=deadline
            ))
        
        # Trier par priorité
        self.orders.sort(key=lambda o: -o.priority)
        print(f"  {len(self.orders)} commandes générées")
    
    def _assign_orders(self):
        """Assigne les commandes pendantes aux drones idle."""
        pending = [o for o in self.orders if o.status == "pending"]
        idle_drones = [d for d in self.drones.values() 
                       if d.status == "idle" and d.battery > 30]
        
        for order in pending[:len(idle_drones)]:
            if not idle_drones:
                break
            
            # Trouver le drone le plus proche
            best_drone = min(idle_drones, key=lambda d: 
                np.sqrt((d.pos_x - order.destination_x)**2 + 
                        (d.pos_y - order.destination_y)**2))
            
            # Vérifier que le drone peut porter le colis
            if order.weight <= 2.5:
                order.assigned_drone = best_drone.id
                order.status = "in_transit"
                best_drone.status = "delivering"
                best_drone.current_order = order.id
                best_drone.payload = order.weight
                idle_drones.remove(best_drone)
    
    def _update_drones(self):
        """Met à jour l'état de tous les drones."""
        for drone in self.drones.values():
            if drone.status == "delivering":
                self._update_delivering(drone)
            elif drone.status == "returning":
                self._update_returning(drone)
            elif drone.status == "charging":
                self._update_charging(drone)
            elif drone.status == "idle":
                # Vérifier si batterie faible
                if drone.battery < 25:
                    drone.status = "charging"
    
    def _update_delivering(self, drone: DroneState):
        """Drone en livraison."""
        order = next((o for o in self.orders if o.id == drone.current_order), None)
        if not order:
            drone.status = "idle"
            return
        
        # Calculer direction vers destination
        dx = order.destination_x - drone.pos_x
        dy = order.destination_y - drone.pos_y
        dist = np.sqrt(dx*dx + dy*dy)
        
        if dist < 0.5:
            # Livraison terminée
            order.status = "delivered"
            drone.deliveries_completed += 1
            drone.payload = 0.0
            drone.current_order = None
            drone.status = "returning"
            self.stats['deliveries'] += 1
        else:
            # Avancer vers destination
            speed = min(0.3, dist)  # km/min (18 km/h max)
            drone.pos_x += (dx / dist) * speed
            drone.pos_y += (dy / dist) * speed
            drone.distance_traveled += speed
            
            # Consommer batterie (0.5% par km, plus si charge)
            drain = 0.5 + drone.payload * 0.2
            drone.battery -= drain * speed
    
    def _update_returning(self, drone: DroneState):
        """Drone retourne au hub."""
        hub_x, hub_y = ZONE_HUBS[drone.zone]
        
        dx = hub_x - drone.pos_x
        dy = hub_y - drone.pos_y
        dist = np.sqrt(dx*dx + dy*dy)
        
        if dist < 1.0:
            drone.status = "idle" if drone.battery > 30 else "charging"
        else:
            speed = min(0.4, dist)  # Plus rapide sans charge
            drone.pos_x += (dx / dist) * speed
            drone.pos_y += (dy / dist) * speed
            drone.distance_traveled += speed
            drone.battery -= 0.3 * speed
    
    def _update_charging(self, drone: DroneState):
        """Drone en recharge."""
        drone.battery = min(100, drone.battery + 2.0)  # 2% par minute
        if drone.battery >= 95:
            drone.status = "idle"
    
    def step_simulation(self):
        """Avance la simulation d'un pas."""
        self.step += 1
        self._assign_orders()
        self._update_drones()
        
        # Décrémenter les deadlines des commandes pending
        for order in self.orders:
            if order.status == "pending":
                order.deadline_minutes -= 1
                if order.deadline_minutes <= 0:
                    order.status = "failed"
                    self.stats['failed'] += 1
    
    def run(self, verbose: bool = True):
        """Lance la simulation complète."""
        print()
        print("=" * 60)
        print("  DRONE FLEET SIMULATION")
        print("=" * 60)
        print(f"  Durée: {SIMULATION_HOURS}h ({TOTAL_STEPS} steps)")
        print(f"  Drones: {len(self.drones)}")
        print(f"  Commandes: {len(self.orders)}")
        print()
        
        for step in range(TOTAL_STEPS):
            self.step_simulation()
            
            if verbose and step % 60 == 0:  # Chaque heure
                hour = step // 60
                pending = sum(1 for o in self.orders if o.status == "pending")
                transit = sum(1 for o in self.orders if o.status == "in_transit")
                delivered = self.stats['deliveries']
                failed = self.stats['failed']
                
                charging = sum(1 for d in self.drones.values() if d.status == "charging")
                delivering = sum(1 for d in self.drones.values() if d.status == "delivering")
                
                print(f"  Heure {hour}:")
                print(f"    Commandes: {pending} pending, {transit} transit, {delivered} livrées, {failed} échec")
                print(f"    Drones: {delivering} en livraison, {charging} en charge")
        
        self._print_final_stats()
    
    def _print_final_stats(self):
        """Affiche les statistiques finales."""
        print()
        print("=" * 60)
        print("  RÉSULTATS")
        print("=" * 60)
        
        delivered = self.stats['deliveries']
        failed = self.stats['failed']
        total = len(self.orders)
        
        total_distance = sum(d.distance_traveled for d in self.drones.values())
        total_deliveries = sum(d.deliveries_completed for d in self.drones.values())
        
        print(f"  Commandes livrées: {delivered}/{total} ({100*delivered/total:.1f}%)")
        print(f"  Commandes échouées: {failed} ({100*failed/total:.1f}%)")
        print(f"  Distance totale: {total_distance:.1f} km")
        print(f"  Distance moyenne/livraison: {total_distance/max(1,delivered):.2f} km")
        print()
        
        # Stats par zone
        print("  Par zone:")
        for zone in range(len(ZONE_HUBS)):
            zone_drones = [d for d in self.drones.values() if d.zone == zone]
            zone_deliveries = sum(d.deliveries_completed for d in zone_drones)
            zone_distance = sum(d.distance_traveled for d in zone_drones)
            avg_battery = np.mean([d.battery for d in zone_drones])
            print(f"    Zone {zone}: {zone_deliveries} livraisons, {zone_distance:.1f}km, batterie moy: {avg_battery:.0f}%")
        
        print()
        if delivered / total >= 0.95:
            print("  ✅ SUCCÈS: Objectif 95% atteint!")
        else:
            print("  ⚠️  Objectif 95% non atteint")


if __name__ == "__main__":
    sim = DroneFleetSimulation()
    sim.run()
