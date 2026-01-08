#!/usr/bin/env python3
"""
AMAZON FULFILLMENT CENTER SIMULATION - Réaliste
================================================

Simulation complète d'un centre de préparation Amazon.
Utilise la config exhaustive amazon_fulfillment.yaml.

Scénario:
- 200 robots Kiva dans 4 zones de picking
- 50 stations de packing
- 10 quais d'expédition
- 5000 commandes à traiter en 8 heures
- Contraintes: batterie robots, capacité convoyeurs, deadlines

Run:
    python examples/amazon_fulfillment_simulation.py
"""

import numpy as np
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from enum import Enum

sys.path.insert(0, '.')

from lifecore import LifeCore, Need, Goal, SharedResource
from lifecore.config import load_system


# === CONFIGURATION SIMULATION ===

SIMULATION_HOURS = 8
STEPS_PER_HOUR = 60  # 1 step = 1 minute
TOTAL_STEPS = SIMULATION_HOURS * STEPS_PER_HOUR
NUM_ORDERS = 5000

# Layout entrepôt (200m x 100m)
WAREHOUSE_WIDTH = 200
WAREHOUSE_HEIGHT = 100

# Zones
PICKING_ZONES = [
    {"id": 0, "x_range": (0, 50), "y_range": (0, 100)},
    {"id": 1, "x_range": (50, 100), "y_range": (0, 100)},
    {"id": 2, "x_range": (100, 150), "y_range": (0, 100)},
    {"id": 3, "x_range": (150, 200), "y_range": (0, 100)},
]

PACKING_AREA = {"x_range": (80, 120), "y_range": (40, 60)}
SHIPPING_DOCKS = {"x_range": (90, 110), "y_range": (90, 100)}
CHARGING_STATIONS = [(25, 50), (75, 50), (125, 50), (175, 50)]


# === DATA STRUCTURES ===

class OrderStatus(Enum):
    PENDING = "pending"
    PICKING = "picking"
    PICKED = "picked"
    PACKING = "packing"
    PACKED = "packed"
    SHIPPING = "shipping"
    SHIPPED = "shipped"
    FAILED = "failed"


@dataclass
class OrderItem:
    """Item dans une commande."""
    sku: str
    location_x: float
    location_y: float
    zone: int
    weight: float  # kg
    picked: bool = False


@dataclass
class Order:
    """Commande client."""
    id: int
    priority: int  # 1-10 (10 = Prime)
    items: List[OrderItem] = field(default_factory=list)
    deadline_minutes: int = 120
    status: OrderStatus = OrderStatus.PENDING
    assigned_robots: List[int] = field(default_factory=list)
    assigned_packing: Optional[int] = None
    creation_time: int = 0
    completion_time: Optional[int] = None


@dataclass
class KivaRobot:
    """Robot Kiva."""
    id: int
    zone: int
    pos_x: float
    pos_y: float
    battery: float
    status: str  # idle, picking, transporting, charging
    current_order: Optional[int] = None
    current_item_idx: int = 0
    carrying_items: List[OrderItem] = field(default_factory=list)
    distance_traveled: float = 0.0
    items_picked: int = 0


@dataclass
class PackingStation:
    """Station de packing."""
    id: int
    pos_x: float
    pos_y: float
    status: str  # idle, packing
    current_order: Optional[int] = None
    efficiency: float = 1.0  # items/minute
    orders_packed: int = 0


@dataclass
class ShippingDock:
    """Quai d'expédition."""
    id: int
    pos_x: float
    pos_y: float
    status: str  # idle, loading
    orders_shipped: int = 0


# === SIMULATION ===

class AmazonFulfillmentSimulation:
    """Simulation réaliste de centre Amazon."""
    
    def __init__(self, config_path: str = "configs/amazon_fulfillment.yaml"):
        print("Chargement de la configuration...")
        self.system = load_system(config_path)
        self.root = self.system['root']
        self.resources = self.system['resources']
        
        self.orders: Dict[int, Order] = {}
        self.robots: Dict[int, KivaRobot] = {}
        self.packing_stations: Dict[int, PackingStation] = {}
        self.shipping_docks: Dict[int, ShippingDock] = {}
        
        self.step = 0
        self.conveyor_queue: List[int] = []  # Orders waiting for packing
        
        # Stats
        self.stats = {
            'orders_created': 0,
            'orders_shipped': 0,
            'orders_failed': 0,
            'items_picked': 0,
            'total_robot_distance': 0.0,
            'avg_fulfillment_time': 0.0,
            'robot_utilization': 0.0,
            'station_utilization': 0.0
        }
        
        self._init_infrastructure()
        self._generate_orders()
    
    def _init_infrastructure(self):
        """Initialise robots, stations, docks."""
        # 200 robots Kiva (50 par zone)
        robot_id = 0
        for zone in PICKING_ZONES:
            for i in range(50):
                x = np.random.uniform(*zone["x_range"])
                y = np.random.uniform(*zone["y_range"])
                self.robots[robot_id] = KivaRobot(
                    id=robot_id,
                    zone=zone["id"],
                    pos_x=x,
                    pos_y=y,
                    battery=np.random.uniform(80, 100),
                    status="idle"
                )
                robot_id += 1
        
        # 50 stations de packing
        for i in range(50):
            x = np.random.uniform(*PACKING_AREA["x_range"])
            y = np.random.uniform(*PACKING_AREA["y_range"])
            self.packing_stations[i] = PackingStation(
                id=i,
                pos_x=x,
                pos_y=y,
                status="idle",
                efficiency=np.random.uniform(0.8, 1.2)
            )
        
        # 10 quais d'expédition
        for i in range(10):
            x = SHIPPING_DOCKS["x_range"][0] + i * 2
            y = np.random.uniform(*SHIPPING_DOCKS["y_range"])
            self.shipping_docks[i] = ShippingDock(
                id=i,
                pos_x=x,
                pos_y=y,
                status="idle"
            )
        
        print(f"  {len(self.robots)} robots Kiva")
        print(f"  {len(self.packing_stations)} stations de packing")
        print(f"  {len(self.shipping_docks)} quais d'expédition")
    
    def _generate_orders(self):
        """Génère des commandes réalistes."""
        # Distribution des commandes dans le temps
        orders_per_hour = NUM_ORDERS / SIMULATION_HOURS
        
        for i in range(NUM_ORDERS):
            # Moment de création (réparti sur les 8h)
            creation_time = int(i / orders_per_hour * 60)
            
            # Nombre d'items (1-5)
            num_items = np.random.randint(1, 6)
            
            # Priorité
            priority_type = np.random.choice(['prime', 'standard', 'economy'], p=[0.3, 0.5, 0.2])
            priority = {'prime': 9, 'standard': 5, 'economy': 2}[priority_type]
            deadline = {'prime': 60, 'standard': 180, 'economy': 480}[priority_type]
            
            # Items
            items = []
            for j in range(num_items):
                zone = np.random.randint(0, 4)
                zone_info = PICKING_ZONES[zone]
                items.append(OrderItem(
                    sku=f"SKU-{i}-{j}",
                    location_x=np.random.uniform(*zone_info["x_range"]),
                    location_y=np.random.uniform(*zone_info["y_range"]),
                    zone=zone,
                    weight=np.random.uniform(0.1, 5.0)
                ))
            
            self.orders[i] = Order(
                id=i,
                priority=priority,
                items=items,
                deadline_minutes=deadline,
                creation_time=creation_time
            )
        
        self.stats['orders_created'] = NUM_ORDERS
        print(f"  {NUM_ORDERS} commandes générées")
    
    def _get_available_orders(self) -> List[Order]:
        """Retourne les commandes disponibles pour picking."""
        return [o for o in self.orders.values() 
                if o.status == OrderStatus.PENDING 
                and o.creation_time <= self.step]
    
    def _assign_picking_tasks(self):
        """Assigne les commandes aux robots."""
        available = sorted(self._get_available_orders(), 
                          key=lambda o: -o.priority)[:100]  # Top 100
        
        idle_robots = [r for r in self.robots.values() 
                      if r.status == "idle" and r.battery > 20]
        
        for order in available:
            if not idle_robots:
                break
            
            # Trouver le robot le plus proche du premier item à picker
            unpicked = [item for item in order.items if not item.picked]
            if not unpicked:
                continue
            
            first_item = unpicked[0]
            best_robot = min(
                [r for r in idle_robots if r.zone == first_item.zone] or idle_robots,
                key=lambda r: np.sqrt((r.pos_x - first_item.location_x)**2 + 
                                     (r.pos_y - first_item.location_y)**2),
                default=None
            )
            
            if best_robot:
                order.status = OrderStatus.PICKING
                order.assigned_robots.append(best_robot.id)
                best_robot.status = "picking"
                best_robot.current_order = order.id
                best_robot.current_item_idx = 0
                idle_robots.remove(best_robot)
    
    def _update_robots(self):
        """Met à jour tous les robots."""
        for robot in self.robots.values():
            if robot.status == "picking":
                self._update_picking_robot(robot)
            elif robot.status == "transporting":
                self._update_transporting_robot(robot)
            elif robot.status == "charging":
                self._update_charging_robot(robot)
            elif robot.status == "idle":
                if robot.battery < 20:
                    robot.status = "charging"
    
    def _update_picking_robot(self, robot: KivaRobot):
        """Robot en picking."""
        if robot.current_order is None:
            robot.status = "idle"
            return
        
        order = self.orders.get(robot.current_order)
        if not order:
            robot.status = "idle"
            return
        
        # Items à picker
        unpicked = [item for item in order.items if not item.picked]
        if not unpicked:
            # Tous les items pickés, transporter vers packing
            robot.status = "transporting"
            return
        
        # Aller vers le prochain item
        target = unpicked[0]
        dx = target.location_x - robot.pos_x
        dy = target.location_y - robot.pos_y
        dist = np.sqrt(dx*dx + dy*dy)
        
        if dist < 1.0:
            # Picker l'item
            target.picked = True
            robot.carrying_items.append(target)
            robot.items_picked += 1
            self.stats['items_picked'] += 1
        else:
            # Avancer (2 m/s = 0.12 km/min)
            speed = min(2.0, dist)
            robot.pos_x += (dx / dist) * speed
            robot.pos_y += (dy / dist) * speed
            robot.distance_traveled += speed
            robot.battery -= 0.1 * speed
    
    def _update_transporting_robot(self, robot: KivaRobot):
        """Robot transporte vers zone packing."""
        # Aller vers le centre de la zone packing
        target_x = (PACKING_AREA["x_range"][0] + PACKING_AREA["x_range"][1]) / 2
        target_y = (PACKING_AREA["y_range"][0] + PACKING_AREA["y_range"][1]) / 2
        
        dx = target_x - robot.pos_x
        dy = target_y - robot.pos_y
        dist = np.sqrt(dx*dx + dy*dy)
        
        if dist < 2.0:
            # Arrivé - déposer sur convoyeur
            if robot.current_order is not None:
                order = self.orders.get(robot.current_order)
                if order:
                    order.status = OrderStatus.PICKED
                    self.conveyor_queue.append(order.id)
            
            robot.carrying_items = []
            robot.current_order = None
            robot.status = "idle"
        else:
            speed = min(2.0, dist)
            robot.pos_x += (dx / dist) * speed
            robot.pos_y += (dy / dist) * speed
            robot.distance_traveled += speed
            robot.battery -= 0.15 * speed  # Plus lourd
    
    def _update_charging_robot(self, robot: KivaRobot):
        """Robot en charge."""
        # Aller vers station de charge la plus proche
        closest = min(CHARGING_STATIONS, 
                     key=lambda s: np.sqrt((s[0] - robot.pos_x)**2 + (s[1] - robot.pos_y)**2))
        
        dx = closest[0] - robot.pos_x
        dy = closest[1] - robot.pos_y
        dist = np.sqrt(dx*dx + dy*dy)
        
        if dist < 1.0:
            # Charger
            robot.battery = min(100, robot.battery + 3.0)
            if robot.battery >= 95:
                robot.status = "idle"
        else:
            speed = min(2.0, dist)
            robot.pos_x += (dx / dist) * speed
            robot.pos_y += (dy / dist) * speed
    
    def _update_packing_stations(self):
        """Met à jour les stations de packing."""
        for station in self.packing_stations.values():
            if station.status == "idle" and self.conveyor_queue:
                # Prendre une commande du convoyeur
                order_id = self.conveyor_queue.pop(0)
                station.current_order = order_id
                station.status = "packing"
                self.orders[order_id].status = OrderStatus.PACKING
                self.orders[order_id].assigned_packing = station.id
            
            elif station.status == "packing":
                order = self.orders.get(station.current_order)
                if order:
                    # Packing prend ~2 min par commande
                    if np.random.random() < 0.5 * station.efficiency:
                        order.status = OrderStatus.PACKED
                        station.orders_packed += 1
                        station.current_order = None
                        station.status = "idle"
    
    def _update_shipping(self):
        """Met à jour l'expédition."""
        packed_orders = [o for o in self.orders.values() if o.status == OrderStatus.PACKED]
        
        for dock in self.shipping_docks.values():
            if dock.status == "idle" and packed_orders:
                order = packed_orders.pop(0)
                order.status = OrderStatus.SHIPPED
                order.completion_time = self.step
                dock.orders_shipped += 1
                self.stats['orders_shipped'] += 1
    
    def _update_deadlines(self):
        """Vérifie les deadlines."""
        for order in self.orders.values():
            if order.status not in [OrderStatus.SHIPPED, OrderStatus.FAILED]:
                time_elapsed = self.step - order.creation_time
                if time_elapsed > order.deadline_minutes:
                    order.status = OrderStatus.FAILED
                    self.stats['orders_failed'] += 1
    
    def step_simulation(self):
        """Avance la simulation d'un pas."""
        self.step += 1
        self._assign_picking_tasks()
        self._update_robots()
        self._update_packing_stations()
        self._update_shipping()
        self._update_deadlines()
    
    def run(self, verbose: bool = True):
        """Lance la simulation."""
        print()
        print("=" * 60)
        print("  AMAZON FULFILLMENT CENTER SIMULATION")
        print("=" * 60)
        print(f"  Durée: {SIMULATION_HOURS}h ({TOTAL_STEPS} steps)")
        print(f"  Robots: {len(self.robots)}")
        print(f"  Commandes: {NUM_ORDERS}")
        print()
        
        for step in range(TOTAL_STEPS):
            self.step_simulation()
            
            if verbose and step % 60 == 0:
                hour = step // 60
                shipped = self.stats['orders_shipped']
                failed = self.stats['orders_failed']
                
                pending = sum(1 for o in self.orders.values() if o.status == OrderStatus.PENDING)
                picking = sum(1 for o in self.orders.values() if o.status == OrderStatus.PICKING)
                packing = sum(1 for o in self.orders.values() if o.status in [OrderStatus.PICKED, OrderStatus.PACKING])
                
                robots_picking = sum(1 for r in self.robots.values() if r.status == "picking")
                robots_charging = sum(1 for r in self.robots.values() if r.status == "charging")
                
                print(f"  Heure {hour}:")
                print(f"    Commandes: {pending} pending, {picking} picking, {packing} packing, {shipped} expédiées, {failed} échec")
                print(f"    Robots: {robots_picking} picking, {robots_charging} charging")
        
        self._print_final_stats()
    
    def _print_final_stats(self):
        """Affiche les statistiques finales."""
        print()
        print("=" * 60)
        print("  RÉSULTATS")
        print("=" * 60)
        
        shipped = self.stats['orders_shipped']
        failed = self.stats['orders_failed']
        total = NUM_ORDERS
        
        # Temps moyen de fulfillment
        completed = [o for o in self.orders.values() if o.completion_time]
        if completed:
            avg_time = np.mean([o.completion_time - o.creation_time for o in completed])
        else:
            avg_time = 0
        
        total_distance = sum(r.distance_traveled for r in self.robots.values())
        total_items = self.stats['items_picked']
        
        print(f"  Commandes expédiées: {shipped}/{total} ({100*shipped/total:.1f}%)")
        print(f"  Commandes échouées: {failed} ({100*failed/total:.1f}%)")
        print(f"  Items pickés: {total_items}")
        print(f"  Distance robots: {total_distance:.1f} m")
        print(f"  Temps moyen fulfillment: {avg_time:.1f} min")
        print()
        
        # Stats par zone
        print("  Par zone picking:")
        for zone in range(4):
            zone_robots = [r for r in self.robots.values() if r.zone == zone]
            zone_items = sum(r.items_picked for r in zone_robots)
            zone_dist = sum(r.distance_traveled for r in zone_robots)
            avg_battery = np.mean([r.battery for r in zone_robots])
            print(f"    Zone {zone}: {zone_items} items, {zone_dist:.1f}m, batterie moy: {avg_battery:.0f}%")
        
        print()
        print("  Stations packing:")
        total_packed = sum(s.orders_packed for s in self.packing_stations.values())
        active = sum(1 for s in self.packing_stations.values() if s.orders_packed > 0)
        print(f"    {total_packed} commandes packées par {active} stations actives")
        
        print()
        print("  Quais expédition:")
        for dock in self.shipping_docks.values():
            if dock.orders_shipped > 0:
                print(f"    Quai {dock.id}: {dock.orders_shipped} expéditions")
        
        print()
        if shipped / total >= 0.95:
            print("  ✅ SUCCÈS: Objectif 95% atteint!")
        else:
            print("  ⚠️  Objectif 95% non atteint")


if __name__ == "__main__":
    np.random.seed(42)  # Reproductibilité
    sim = AmazonFulfillmentSimulation()
    sim.run()
