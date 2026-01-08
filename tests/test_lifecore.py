#!/usr/bin/env python3
"""Tests pour LifeCore V3."""

import numpy as np
import sys
sys.path.insert(0, '.')

from lifecore import LifeCore, Need, TensorMemory, create_homeostatic_need


def test_memory_reuse():
    """Test que la mémoire est réutilisée."""
    agent = LifeCore(dims=4)
    state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    good_intent = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    
    # Avant apprentissage → intention vide (pas de besoins)
    intent_before = agent.get_intention(state)
    assert np.allclose(intent_before, 0), "Devrait être vide sans besoins"
    
    # Ajouter un besoin pour avoir de l'urgence
    agent.needs.append(create_homeostatic_need(0, 4, target_value=0.5))
    
    # Apprendre
    agent.add_experience(state, good_intent, good_intent * 0.1, quality=0.9)
    
    # Après apprentissage → réutilise
    intent_after = agent.get_intention(state)
    assert np.allclose(intent_after, good_intent), f"Devrait réutiliser: {intent_after}"
    print("✓ test_memory_reuse PASS")


def test_needs_generate_intention():
    """Test que les besoins génèrent des intentions."""
    energy_need = create_homeostatic_need(
        target_dim=0, dims=4, target_value=1.0, priority=1.0
    )
    agent = LifeCore(dims=4, needs=[energy_need])
    
    # État avec énergie basse
    low_energy_state = np.array([0.2, 0.5, 0.0, 0.0], dtype=np.float32)
    intent = agent.get_intention(low_energy_state)
    
    # L'intention devrait pousser vers dimension 0
    assert intent[0] > 0, f"Devrait pousser vers dim 0: {intent}"
    print("✓ test_needs_generate_intention PASS")


def test_fractal_memory_sharing():
    """Test que les enfants partagent la mémoire du parent."""
    parent = LifeCore(dims=4)
    child = parent.spawn_child(share_memory=True)
    
    state = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    good_intent = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    
    # Parent apprend
    parent.add_experience(state, good_intent, good_intent * 0.1, quality=0.9)
    
    # Ajouter un besoin à l'enfant pour avoir de l'urgence
    child.needs.append(create_homeostatic_need(0, 4, target_value=0.5))
    
    # L'enfant devrait voir l'expérience
    child_intent = child.get_intention(state)
    assert np.allclose(child_intent, good_intent), "Enfant devrait réutiliser mémoire parent"
    print("✓ test_fractal_memory_sharing PASS")


def test_tensor_memory_query():
    """Test direct de TensorMemory."""
    memory = TensorMemory(max_size=100)
    
    state = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    intent = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    
    memory.add(state, intent, intent * 0.1, quality=0.9)
    
    result = memory.query(state, threshold=0.7)
    assert result is not None, "Devrait trouver"
    assert np.allclose(result, intent), "Devrait retourner l'intention"
    print("✓ test_tensor_memory_query PASS")


if __name__ == "__main__":
    test_memory_reuse()
    test_needs_generate_intention()
    test_fractal_memory_sharing()
    test_tensor_memory_query()
    print()
    print("=" * 40)
    print("✅ TOUS LES TESTS PASSENT")
    print("=" * 40)
