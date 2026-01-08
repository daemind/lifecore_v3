#!/usr/bin/env python3
"""
Démonstration: Goals, Needs, Resources
======================================

Une fusée avec ressources limitées:
- Propulsion (haute priorité) → consomme beaucoup d'énergie
- Navigation (haute priorité) → consomme peu
- Confort (basse priorité) → ne reçoit presque rien

Run:
    cd lifecore-v3-clean
    python examples/rocket_resources.py
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from lifecore import LifeCore, Need, Goal, SharedResource, create_homeostatic_need


def run_rocket_demo():
    print("=" * 60)
    print("  FUSÉE: Goals, Needs, Resources")
    print("=" * 60)
    print()
    
    # === Ressource partagée: Énergie ===
    energy = SharedResource("energy", capacity=100.0)
    print(f"Énergie disponible: {energy.capacity} unités")
    print()
    
    # === Créer la fusée (LifeCore racine) ===
    rocket = LifeCore(dims=4)
    rocket.state = np.array([0.0, 0.0, 0.0, 100.0])  # [x, y, vel, fuel]
    
    # Goal: atteindre l'orbite
    orbit_target = np.array([0.0, 100.0, 0.0, 0.0])
    rocket.goals.push(Goal(target=orbit_target, name="reach_orbit", priority=10.0))
    
    # === Sous-systèmes (enfants) ===
    
    # Propulsion: haute priorité, consomme beaucoup
    propulsion = rocket.spawn_child(share_memory=True)
    propulsion.add_resource(energy, priority=10.0)  # Priorité maximale
    propulsion_need = create_homeostatic_need(
        target_dim=1, dims=4, target_value=100.0, priority=10.0, name="thrust"
    )
    propulsion.needs.append(propulsion_need)
    
    # Navigation: haute priorité, consomme peu
    navigation = rocket.spawn_child(share_memory=True)
    navigation.add_resource(energy, priority=8.0)
    nav_need = create_homeostatic_need(
        target_dim=0, dims=4, target_value=0.0, priority=8.0, name="guidance"
    )
    navigation.needs.append(nav_need)
    
    # Confort (vie des astronautes): basse priorité
    comfort = rocket.spawn_child(share_memory=True)
    comfort.add_resource(energy, priority=1.0)  # Priorité minimale
    comfort_need = create_homeostatic_need(
        target_dim=3, dims=4, target_value=20.0, priority=1.0, name="life_support"
    )
    comfort.needs.append(comfort_need)
    
    print("Sous-systèmes:")
    print(f"  Propulsion  (priorité 10) → besoin: atteindre altitude")
    print(f"  Navigation  (priorité 8)  → besoin: garder trajectoire")
    print(f"  Confort     (priorité 1)  → besoin: maintenir vie")
    print()
    
    # === Simulation ===
    print("Allocation des ressources:")
    
    # Chaque sous-système demande de l'énergie
    propulsion_demand = 60.0
    navigation_demand = 20.0
    comfort_demand = 40.0
    
    print(f"  Propulsion demande {propulsion_demand} unités")
    print(f"  Navigation demande {navigation_demand} unités")
    print(f"  Confort demande {comfort_demand} unités")
    print(f"  Total demandé: {propulsion_demand + navigation_demand + comfort_demand}")
    print()
    
    # Faire les demandes
    energy.request(propulsion, propulsion_demand)
    energy.request(navigation, navigation_demand)
    energy.request(comfort, comfort_demand)
    
    # Voir les allocations
    prop_alloc = energy.get_allocation(propulsion)
    nav_alloc = energy.get_allocation(navigation)
    comfort_alloc = energy.get_allocation(comfort)
    
    print("Résultat de l'allocation (selon priorités):")
    print(f"  ⚡ Propulsion: {prop_alloc:.1f} / {propulsion_demand} ({prop_alloc/propulsion_demand*100:.0f}%)")
    print(f"  🧭 Navigation: {nav_alloc:.1f} / {navigation_demand} ({nav_alloc/navigation_demand*100:.0f}%)")
    print(f"  🛋️  Confort:    {comfort_alloc:.1f} / {comfort_demand} ({comfort_alloc/comfort_demand*100:.0f}%)")
    print()
    
    # Vérifier que le total ne dépasse pas la capacité
    total_alloc = prop_alloc + nav_alloc + comfort_alloc
    print(f"Total alloué: {total_alloc:.1f} / {energy.capacity}")
    print(f"Utilisation: {energy.utilization()*100:.0f}%")
    print()
    
    # === Démonstration de l'émergence ===
    print("=" * 60)
    print("  ÉMERGENCE: 'Pourquoi pas de fenêtres?'")
    print("=" * 60)
    print()
    
    # Ajouter un sous-système "fenêtres" (très basse priorité)
    windows = rocket.spawn_child(share_memory=True)
    windows.add_resource(energy, priority=0.1)  # Priorité négligeable
    
    # Les fenêtres demandent de l'énergie
    windows_demand = 10.0
    energy.request(windows, windows_demand)
    windows_alloc = energy.get_allocation(windows)
    
    print(f"Fenêtres (priorité 0.1) demandent {windows_demand} unités")
    print(f"Fenêtres reçoivent: {windows_alloc:.2f} unités ({windows_alloc/windows_demand*100:.1f}%)")
    print()
    
    if windows_alloc < 1.0:
        print("🚫 Les fenêtres n'ont pas assez de ressources pour exister!")
        print("   → C'est pour ça qu'il n'y a pas de fenêtres sur une fusée.")
    else:
        print("✓ Les fenêtres ont assez de ressources.")
    
    print()
    print("Conclusion: La hiérarchie des priorités fait ÉMERGER")
    print("            la structure finale du système.")


if __name__ == "__main__":
    run_rocket_demo()
