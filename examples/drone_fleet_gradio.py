#!/usr/bin/env python3
"""
DRONE FLEET VISUALIZATION - Gradio Dashboard
=============================================

Animation interactive de la flotte de drones.
Visualise en temps réel: positions, états, livraisons.

Run:
    python examples/drone_fleet_gradio.py
    
Puis ouvrir: http://localhost:7860
"""

import numpy as np
import gradio as gr
import time
from dataclasses import dataclass
from typing import List, Dict, Optional
import sys

sys.path.insert(0, '.')

from lifecore.config import load_system


# === CONFIGURATION ===

NUM_DRONES = 50  # Réduit pour visualisation
NUM_ORDERS = 200
SIMULATION_SPEED = 0.1  # Secondes par step

ZONE_HUBS = [
    (10, 10), (40, 10),
    (10, 40), (40, 40)
]

NO_FLY_ZONES = [
    {"center": (25, 25), "radius": 5, "name": "Aéroport"},
    {"center": (45, 5), "radius": 3, "name": "Zone militaire"},
]


# === DATA STRUCTURES ===

@dataclass
class Drone:
    id: int
    x: float
    y: float
    battery: float
    status: str  # idle, delivering, returning, charging
    target_x: Optional[float] = None
    target_y: Optional[float] = None
    deliveries: int = 0


@dataclass
class Order:
    id: int
    dest_x: float
    dest_y: float
    status: str  # pending, in_transit, delivered


# === SIMULATION STATE ===

class DroneFleetVisualization:
    
    def __init__(self):
        self.drones: Dict[int, Drone] = {}
        self.orders: List[Order] = []
        self.step = 0
        self.running = False
        
        self._init()
    
    def _init(self):
        """Initialise la simulation."""
        # Drones
        drone_id = 0
        for hub_x, hub_y in ZONE_HUBS:
            for i in range(NUM_DRONES // len(ZONE_HUBS)):
                self.drones[drone_id] = Drone(
                    id=drone_id,
                    x=hub_x + np.random.uniform(-3, 3),
                    y=hub_y + np.random.uniform(-3, 3),
                    battery=np.random.uniform(80, 100),
                    status="idle"
                )
                drone_id += 1
        
        # Commandes
        for i in range(NUM_ORDERS):
            self.orders.append(Order(
                id=i,
                dest_x=np.random.uniform(5, 45),
                dest_y=np.random.uniform(5, 45),
                status="pending"
            ))
    
    def step_simulation(self):
        """Avance d'un pas."""
        self.step += 1
        
        # Assigner les commandes pending
        pending = [o for o in self.orders if o.status == "pending"]
        idle_drones = [d for d in self.drones.values() if d.status == "idle" and d.battery > 25]
        
        for order in pending[:len(idle_drones)]:
            if not idle_drones:
                break
            
            # Drone le plus proche
            drone = min(idle_drones, key=lambda d: 
                np.sqrt((d.x - order.dest_x)**2 + (d.y - order.dest_y)**2))
            
            order.status = "in_transit"
            drone.status = "delivering"
            drone.target_x = order.dest_x
            drone.target_y = order.dest_y
            idle_drones.remove(drone)
        
        # Mettre à jour les drones
        for drone in self.drones.values():
            if drone.status == "delivering":
                self._move_drone(drone)
            elif drone.status == "returning":
                self._return_drone(drone)
            elif drone.status == "charging":
                drone.battery = min(100, drone.battery + 2)
                if drone.battery > 90:
                    drone.status = "idle"
            elif drone.status == "idle" and drone.battery < 20:
                drone.status = "charging"
    
    def _move_drone(self, drone: Drone):
        """Déplace le drone vers sa cible."""
        if drone.target_x is None:
            drone.status = "idle"
            return
        
        dx = drone.target_x - drone.x
        dy = drone.target_y - drone.y
        dist = np.sqrt(dx*dx + dy*dy)
        
        if dist < 1:
            # Livraison terminée
            drone.deliveries += 1
            drone.status = "returning"
            drone.target_x = ZONE_HUBS[drone.id % len(ZONE_HUBS)][0]
            drone.target_y = ZONE_HUBS[drone.id % len(ZONE_HUBS)][1]
            
            # Marquer la commande comme livrée
            for order in self.orders:
                if order.status == "in_transit":
                    order.status = "delivered"
                    break
        else:
            speed = min(1.5, dist)
            drone.x += (dx / dist) * speed
            drone.y += (dy / dist) * speed
            drone.battery -= 0.3
    
    def _return_drone(self, drone: Drone):
        """Retourne au hub."""
        dx = drone.target_x - drone.x
        dy = drone.target_y - drone.y
        dist = np.sqrt(dx*dx + dy*dy)
        
        if dist < 2:
            drone.status = "idle"
            drone.target_x = None
            drone.target_y = None
        else:
            speed = min(2.0, dist)
            drone.x += (dx / dist) * speed
            drone.y += (dy / dist) * speed
            drone.battery -= 0.2
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques."""
        delivered = sum(1 for o in self.orders if o.status == "delivered")
        pending = sum(1 for o in self.orders if o.status == "pending")
        in_transit = sum(1 for o in self.orders if o.status == "in_transit")
        
        delivering = sum(1 for d in self.drones.values() if d.status == "delivering")
        returning = sum(1 for d in self.drones.values() if d.status == "returning")
        charging = sum(1 for d in self.drones.values() if d.status == "charging")
        idle = sum(1 for d in self.drones.values() if d.status == "idle")
        
        avg_battery = np.mean([d.battery for d in self.drones.values()])
        
        return {
            "step": self.step,
            "delivered": delivered,
            "pending": pending,
            "in_transit": in_transit,
            "delivering": delivering,
            "returning": returning,
            "charging": charging,
            "idle": idle,
            "avg_battery": avg_battery,
            "total_deliveries": sum(d.deliveries for d in self.drones.values())
        }
    
    def render(self) -> str:
        """Génère le rendu HTML de la carte."""
        width, height = 500, 500
        scale = 10
        
        svg_parts = [
            f'<svg width="{width}" height="{height}" style="background: #1a1a2e; border-radius: 10px;">',
            # Grille
            '<g stroke="#333" stroke-width="0.5">',
        ]
        
        for i in range(0, 51, 10):
            svg_parts.append(f'<line x1="{i*scale}" y1="0" x2="{i*scale}" y2="{height}"/>')
            svg_parts.append(f'<line x1="0" y1="{i*scale}" x2="{width}" y2="{i*scale}"/>')
        svg_parts.append('</g>')
        
        # Zones interdites
        for zone in NO_FLY_ZONES:
            cx, cy = zone["center"]
            r = zone["radius"] * scale
            svg_parts.append(
                f'<circle cx="{cx*scale}" cy="{cy*scale}" r="{r}" '
                f'fill="rgba(255,0,0,0.2)" stroke="red" stroke-width="2"/>'
            )
            svg_parts.append(
                f'<text x="{cx*scale}" y="{cy*scale}" fill="red" text-anchor="middle" '
                f'font-size="10">{zone["name"]}</text>'
            )
        
        # Hubs
        for i, (hx, hy) in enumerate(ZONE_HUBS):
            svg_parts.append(
                f'<rect x="{hx*scale-15}" y="{hy*scale-15}" width="30" height="30" '
                f'fill="rgba(0,255,0,0.2)" stroke="green" stroke-width="2" rx="5"/>'
            )
            svg_parts.append(
                f'<text x="{hx*scale}" y="{hy*scale+5}" fill="green" text-anchor="middle" '
                f'font-size="12">Hub {i}</text>'
            )
        
        # Destinations (commandes in_transit)
        for order in self.orders:
            if order.status == "in_transit":
                svg_parts.append(
                    f'<circle cx="{order.dest_x*scale}" cy="{order.dest_y*scale}" r="5" '
                    f'fill="yellow" opacity="0.7"/>'
                )
        
        # Drones
        for drone in self.drones.values():
            x, y = drone.x * scale, drone.y * scale
            
            # Couleur selon statut
            colors = {
                "idle": "#00ff00",
                "delivering": "#ffff00",
                "returning": "#00ffff",
                "charging": "#ff0000"
            }
            color = colors.get(drone.status, "#ffffff")
            
            # Forme selon batterie
            size = 6 + (drone.battery / 100) * 4
            
            # Drone
            svg_parts.append(
                f'<circle cx="{x}" cy="{y}" r="{size}" fill="{color}" opacity="0.9"/>'
            )
            
            # Ligne vers target si en mission
            if drone.target_x and drone.status == "delivering":
                svg_parts.append(
                    f'<line x1="{x}" y1="{y}" '
                    f'x2="{drone.target_x*scale}" y2="{drone.target_y*scale}" '
                    f'stroke="{color}" stroke-width="1" opacity="0.4"/>'
                )
        
        svg_parts.append('</svg>')
        return ''.join(svg_parts)
    
    def reset(self):
        """Réinitialise la simulation."""
        self.drones.clear()
        self.orders.clear()
        self.step = 0
        self._init()


# === GRADIO APP ===

sim = DroneFleetVisualization()

def step_and_render():
    """Avance d'un step et retourne le rendu."""
    sim.step_simulation()
    stats = sim.get_stats()
    
    status_text = f"""
**Step:** {stats['step']}

**Commandes:**
- 📦 En attente: {stats['pending']}
- 🚚 En transit: {stats['in_transit']}
- ✅ Livrées: {stats['delivered']}

**Drones:**
- 🟢 Idle: {stats['idle']}
- 🟡 En livraison: {stats['delivering']}
- 🔵 Retour: {stats['returning']}
- 🔴 En charge: {stats['charging']}

**Batterie moyenne:** {stats['avg_battery']:.1f}%
**Total livraisons:** {stats['total_deliveries']}
    """
    
    return sim.render(), status_text

def run_simulation(steps: int):
    """Lance N steps de simulation."""
    for _ in range(steps):
        yield step_and_render()

def reset_simulation():
    """Réinitialise tout."""
    sim.reset()
    return sim.render(), "Simulation réinitialisée"

# Créer l'interface
with gr.Blocks(title="Drone Fleet Simulation") as app:
    gr.Markdown("# 🚁 Drone Fleet Simulation")
    gr.Markdown("Visualisation en temps réel de la flotte de drones")
    
    with gr.Row():
        with gr.Column(scale=2):
            map_output = gr.HTML(value=sim.render(), label="Carte")
        
        with gr.Column(scale=1):
            stats_output = gr.Markdown(value="Cliquez 'Step' pour commencer")
            
            with gr.Row():
                step_btn = gr.Button("▶️ Step", variant="primary")
                run_btn = gr.Button("⏩ Run 50 steps")
                reset_btn = gr.Button("🔄 Reset")
            
            gr.Markdown("""
            ### Légende
            - 🟢 Vert: Hubs / Drones idle
            - 🟡 Jaune: Drones en livraison / Destinations
            - 🔵 Cyan: Drones en retour
            - 🔴 Rouge: Zones interdites / Drones en charge
            """)
    
    step_btn.click(step_and_render, outputs=[map_output, stats_output])
    run_btn.click(lambda: list(run_simulation(50))[-1], outputs=[map_output, stats_output])
    reset_btn.click(reset_simulation, outputs=[map_output, stats_output])


if __name__ == "__main__":
    print("Lancement du dashboard Gradio...")
    print("Ouvrir: http://localhost:7860")
    app.launch()
