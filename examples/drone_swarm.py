#!/usr/bin/env python3
"""
Exemple: Swarm de Drones avec LifeCore V3
=========================================

Architecture fractale:
    Controller (LifeCore)
        └── Drones (enfants, mémoire partagée)
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from lifecore import LifeCore, Need


class DynamicTargetNeed(Need):
    """Besoin d'atteindre une cible (direction dynamique)."""
    
    def __init__(self, target: np.ndarray, dims: int):
        self.target = target
        super().__init__(
            sub_matrix=np.ones(dims) / np.sqrt(dims),
            extractor=lambda s: float(np.linalg.norm(s[:2] - target[:2])),
            urgency_fn=lambda d: 0.0 if d < 1.0 else 0.5,
            priority=1.0,
            name="target"
        )
    
    def compute_intention(self, state: np.ndarray) -> np.ndarray:
        direction = self.target[:len(state)] - state
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return np.zeros_like(state)
        direction = direction / dist
        urgency = self.urgency_fn(dist)
        return direction * urgency * self.priority


def run_swarm():
    print("=" * 50)
    print("  DRONE SWARM - LifeCore V3")
    print("=" * 50)
    
    # Configuration
    n_drones = 5
    dims = 4
    target = np.array([10.0, 10.0, 0.0, 0.0])
    
    # Controller
    controller = LifeCore(dims=dims)
    
    # Créer les drones
    drones = []
    positions = []
    for i in range(n_drones):
        pos = np.random.randn(dims).astype(np.float32) * 2
        positions.append(pos)
        
        drone = controller.spawn_child([DynamicTargetNeed(target, dims)], share_memory=True)
        drone.similarity_threshold = 0.95
        drone.min_quality = 0.7
        drones.append(drone)
    
    print(f"Swarm: {n_drones} drones, cible: {target[:2]}")
    
    # Simulation
    for step in range(100):
        for i, (drone, pos) in enumerate(zip(drones, positions)):
            intention = drone.get_intention(pos)
            
            # Normaliser
            norm = np.linalg.norm(intention)
            if norm > 1.0:
                intention = intention / norm
            
            # Mouvement
            effect = intention * 0.5
            new_pos = pos + effect
            
            # Apprendre si amélioration
            d_before = np.linalg.norm(pos[:2] - target[:2])
            d_after = np.linalg.norm(new_pos[:2] - target[:2])
            improvement = d_before - d_after
            quality = float(np.clip(0.5 + improvement, 0.1, 1.0))
            
            if quality > 0.5:
                drone.add_experience(pos.copy(), intention.copy(), effect.copy(), quality)
            
            positions[i] = new_pos
        
        if step % 30 == 29:
            distances = [np.linalg.norm(p[:2] - target[:2]) for p in positions]
            print(f"  Step {step+1}: distance moyenne = {np.mean(distances):.2f}")
    
    # Résultat
    final_distances = [np.linalg.norm(p[:2] - target[:2]) for p in positions]
    converged = sum(1 for d in final_distances if d < 2.0)
    
    print()
    print(f"Convergés: {converged}/{n_drones}")
    print(f"Mémoire: {controller.memory.size} expériences")
    
    if converged >= n_drones // 2:
        print("✅ SUCCÈS")
    else:
        print("⚠️ Convergence partielle")


if __name__ == "__main__":
    run_swarm()
