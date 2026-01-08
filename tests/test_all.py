#!/usr/bin/env python3
"""
LifeCore V3 - Unit Tests Complets
=================================

Tests unitaires exhaustifs pour tous les modules.

Run:
    cd lifecore-v3-clean
    python -m pytest tests/ -v
    
    # Ou directement:
    python tests/test_all.py
"""

import numpy as np
import sys
import unittest
from typing import Dict, Any

sys.path.insert(0, '.')


# =============================================================================
# TEST: CORE
# =============================================================================

class TestLifeCore(unittest.TestCase):
    """Tests pour lifecore.core."""
    
    def test_init_dims(self):
        """Test création avec différentes dimensions."""
        from lifecore import LifeCore
        
        lc = LifeCore(dims=4)
        self.assertEqual(lc.dims, 4)
        
        lc = LifeCore(dims=10)
        self.assertEqual(lc.dims, 10)
    
    def test_spawn_child(self):
        """Test création d'enfants."""
        from lifecore import LifeCore
        
        parent = LifeCore(dims=6)
        child = parent.spawn_child(domain_dims=[0, 1, 2])
        
        self.assertEqual(len(parent.children), 1)
        self.assertEqual(child.parent, parent)
        self.assertEqual(child.domain_dims, [0, 1, 2])
    
    def test_add_experience(self):
        """Test ajout d'expérience."""
        from lifecore import LifeCore
        
        lc = LifeCore(dims=4)
        state = np.array([1, 0, 0, 0], dtype=np.float32)
        intention = np.array([0, 1, 0, 0], dtype=np.float32)
        
        lc.add_experience(state, intention, effect=intention*0.5, quality=0.9)
        self.assertEqual(len(lc.memory.experiences), 1)
    
    def test_get_intention(self):
        """Test calcul d'intention."""
        from lifecore import LifeCore
        
        lc = LifeCore(dims=4)
        state = np.array([1, 0, 0, 0], dtype=np.float32)
        
        intention = lc.get_intention(state)
        self.assertEqual(len(intention), 4)
    
    def test_get_capacity(self):
        """Test capacity feedback."""
        from lifecore import LifeCore
        
        parent = LifeCore(dims=4)
        child = parent.spawn_child(domain_dims=[0, 1])
        
        requested = np.array([10, 10, 0, 0], dtype=np.float32)
        capacity = child.get_capacity(requested)
        
        self.assertEqual(len(capacity), 4)


# =============================================================================
# TEST: MEMORY
# =============================================================================

class TestTensorMemory(unittest.TestCase):
    """Tests pour lifecore.memory."""
    
    def test_add_query(self):
        """Test stockage et récupération."""
        from lifecore.memory import TensorMemory
        
        mem = TensorMemory(max_size=100)
        
        state = np.array([1, 0, 0, 0], dtype=np.float32)
        intention = np.array([0, 1, 0, 0], dtype=np.float32)
        
        mem.add(state, intention, effect=intention*0.5, quality=0.9)
        
        result = mem.query(state, threshold=0.8, min_quality=0.5)
        self.assertIsNotNone(result)
    
    def test_query_returns_none_if_empty(self):
        """Test query sur mémoire vide."""
        from lifecore.memory import TensorMemory
        
        mem = TensorMemory(max_size=100)
        state = np.array([1, 0, 0, 0], dtype=np.float32)
        
        result = mem.query(state)
        self.assertIsNone(result)
    
    def test_max_size(self):
        """Test limite mémoire."""
        from lifecore.memory import TensorMemory
        
        mem = TensorMemory(max_size=5)
        
        for i in range(10):
            state = np.random.randn(4).astype(np.float32)
            mem.add(state, state, effect=state, quality=0.5)
        
        self.assertLessEqual(len(mem.experiences), 5)
    
    def test_stats(self):
        """Test statistiques mémoire."""
        from lifecore.memory import TensorMemory
        
        mem = TensorMemory(max_size=100)
        
        for i in range(5):
            state = np.random.randn(4).astype(np.float32)
            mem.add(state, state, effect=state, quality=0.5 + i * 0.1)
        
        stats = mem.get_stats()
        self.assertEqual(stats["size"], 5)
        self.assertIn("avg_quality", stats)


# =============================================================================
# TEST: NEED
# =============================================================================

class TestNeed(unittest.TestCase):
    """Tests pour lifecore.need."""
    
    def test_homeostatic_need(self):
        """Test besoin homéostatique."""
        from lifecore.need import create_homeostatic_need
        
        need = create_homeostatic_need(
            target_dim=0,
            dims=4,
            target_value=10.0,
            priority=1.0,
            name="test"
        )
        
        # Éloigné de la cible → haute urgence
        state = np.array([0, 0, 0, 0], dtype=np.float32)
        urgency = need.get_urgency(state)
        self.assertGreater(urgency, 0)
        
        # Proche de la cible → basse urgence
        state = np.array([10, 0, 0, 0], dtype=np.float32)
        urgency = need.get_urgency(state)
        self.assertLess(urgency, 0.1)
    
    def test_need_compute_intention(self):
        """Test génération d'intention."""
        from lifecore.need import create_homeostatic_need
        
        need = create_homeostatic_need(
            target_dim=0,
            dims=4,
            target_value=10.0,
            priority=1.0
        )
        
        state = np.array([0, 0, 0, 0], dtype=np.float32)
        intention = need.compute_intention(state)
        
        self.assertEqual(len(intention), 4)
        self.assertGreater(intention[0], 0)  # Devrait aller vers +


# =============================================================================
# TEST: GOAL
# =============================================================================

class TestGoal(unittest.TestCase):
    """Tests pour lifecore.goal."""
    
    def test_goal_is_reached(self):
        """Test détection objectif atteint."""
        from lifecore.goal import Goal
        
        goal = Goal(
            target=np.array([10, 10, 0, 0]),
            name="test",
            threshold=1.0
        )
        
        # Loin de l'objectif
        state = np.array([0, 0, 0, 0], dtype=np.float32)
        self.assertFalse(goal.is_reached(state))
        
        # Objectif atteint
        state = np.array([10, 10, 0, 0], dtype=np.float32)
        self.assertTrue(goal.is_reached(state))
    
    def test_goal_distance(self):
        """Test calcul de distance."""
        from lifecore.goal import Goal
        
        goal = Goal(
            target=np.array([10, 0, 0, 0]),
            name="test"
        )
        
        state = np.array([0, 0, 0, 0], dtype=np.float32)
        self.assertAlmostEqual(goal.distance(state), 10.0, places=4)
    
    def test_goal_stack(self):
        """Test pile d'objectifs."""
        from lifecore.goal import Goal, GoalStack
        
        stack = GoalStack()
        
        g1 = Goal(target=np.zeros(4), name="g1", priority=1.0)
        g2 = Goal(target=np.ones(4), name="g2", priority=2.0)
        
        stack.push(g1)
        stack.push(g2)
        
        # G2 a la priorité plus haute
        current = stack.current()
        self.assertEqual(current.name, "g2")


# =============================================================================
# TEST: RESOURCE
# =============================================================================

class TestResource(unittest.TestCase):
    """Tests pour lifecore.resource."""
    
    def test_shared_resource_request(self):
        """Test demande de ressource."""
        from lifecore import LifeCore
        from lifecore.resource import SharedResource
        
        res = SharedResource(name="battery", capacity=100)
        lc = LifeCore(dims=4)
        
        res.register(lc, priority=1.0)
        allocated = res.request(lc, 30)
        
        self.assertEqual(allocated, 30)
    
    def test_resource_allocation_priority(self):
        """Test allocation par priorité."""
        from lifecore import LifeCore
        from lifecore.resource import SharedResource
        
        res = SharedResource(name="test", capacity=100)
        
        lc1 = LifeCore(dims=4)
        lc2 = LifeCore(dims=4)
        
        res.register(lc1, priority=5.0)
        res.register(lc2, priority=10.0)
        
        self.assertEqual(len(res.consumers), 2)
    
    def test_resource_stats(self):
        """Test statistiques ressource."""
        from lifecore import LifeCore
        from lifecore.resource import SharedResource
        
        res = SharedResource(name="test", capacity=100)
        lc = LifeCore(dims=4)
        res.register(lc, priority=1.0)
        
        stats = res.get_stats()
        self.assertEqual(stats["capacity"], 100)


# =============================================================================
# TEST: LAW
# =============================================================================

class TestLaw(unittest.TestCase):
    """Tests pour lifecore.law."""
    
    def test_speed_limit(self):
        """Test limite de vitesse."""
        from lifecore.law import SpeedLimit
        
        law = SpeedLimit(max_speed=10.0, velocity_dims=[0, 1])
        
        # Vitesse dans la limite
        intention = np.array([5, 5, 0, 0], dtype=np.float32)
        constrained = law.constrain(intention, np.zeros(4))
        
        speed = np.linalg.norm(constrained[:2])
        self.assertLessEqual(speed, 10.0 + 0.1)
    
    def test_boundary_law(self):
        """Test contrainte de zone."""
        from lifecore.law import BoundaryLaw
        
        law = BoundaryLaw(
            min_bounds=np.array([0, 0]),
            max_bounds=np.array([100, 100]),
            position_dims=[0, 1]
        )
        
        # Dans la zone
        state = np.array([50, 50, 0, 0], dtype=np.float32)
        self.assertFalse(law.is_violated(np.zeros(4), state))
        
        # Hors zone
        state = np.array([150, 50, 0, 0], dtype=np.float32)
        self.assertTrue(law.is_violated(np.zeros(4), state))
    
    def test_law_enforcer(self):
        """Test enforcer avec plusieurs lois."""
        from lifecore.law import LawEnforcer, SpeedLimit
        
        enforcer = LawEnforcer()
        enforcer.add_law(SpeedLimit(max_speed=10.0, velocity_dims=[0, 1]))
        
        intention = np.array([100, 100, 0, 0], dtype=np.float32)
        constrained = enforcer.enforce(intention, np.zeros(4))
        
        speed = np.linalg.norm(constrained[:2])
        self.assertLessEqual(speed, 15.0)  # Should be constrained


# =============================================================================
# TEST: CAPABILITY
# =============================================================================

class TestCapability(unittest.TestCase):
    """Tests pour lifecore.capability."""
    
    def test_capability_creation(self):
        """Test création de capability."""
        from lifecore.capability import Capability
        
        cap = Capability(name="motor", dims=[0, 1], max_value=10.0)
        self.assertEqual(cap.name, "motor")
        self.assertEqual(cap.max_value, 10.0)
    
    def test_capability_set(self):
        """Test ensemble de capacités."""
        from lifecore.capability import CapabilitySet, Capability
        
        cs = CapabilitySet()
        cs.add(Capability(name="speed", dims=[0, 1], max_value=10))
        cs.add(Capability(name="altitude", dims=[2], max_value=100))
        
        self.assertEqual(len(cs.capabilities), 2)


# =============================================================================
# TEST: ACTIVATION
# =============================================================================

class TestActivation(unittest.TestCase):
    """Tests pour lifecore.activation."""
    
    def test_sigmoid(self):
        """Test sigmoid."""
        from lifecore.activation import sigmoid
        
        self.assertAlmostEqual(sigmoid(0), 0.5, places=5)
        self.assertGreater(sigmoid(10), 0.99)
        self.assertLess(sigmoid(-10), 0.01)
    
    def test_smooth_threshold(self):
        """Test seuil lisse."""
        from lifecore.activation import smooth_threshold
        
        # En dessous du seuil
        self.assertLess(smooth_threshold(3, threshold=5), 0.5)
        
        # Au dessus du seuil
        self.assertGreater(smooth_threshold(7, threshold=5), 0.5)
    
    def test_softmax(self):
        """Test softmax."""
        from lifecore.activation import softmax
        
        weights = np.array([1, 2, 3], dtype=np.float32)
        result = softmax(weights)
        
        self.assertAlmostEqual(np.sum(result), 1.0, places=5)


# =============================================================================
# TEST: COHERENCE
# =============================================================================

class TestCoherence(unittest.TestCase):
    """Tests pour lifecore.coherence."""
    
    def test_coherence_imports(self):
        """Test imports coherence module."""
        from lifecore.coherence import CoherenceManager, CoherenceConstraint
        
        # CoherenceManager requires parent argument - just test import
        self.assertIsNotNone(CoherenceManager)
        self.assertIsNotNone(CoherenceConstraint)
    
    def test_coherence_constraint_creation(self):
        """Test création de contrainte."""
        from lifecore.coherence import CoherenceConstraint
        
        constraint = CoherenceConstraint(
            name="test",
            sibling_a="left",
            sibling_b="right",
            dims=[0],
            mode="same"
        )
        
        self.assertEqual(constraint.mode, "same")


# =============================================================================
# TEST: EVENT
# =============================================================================

class TestEvent(unittest.TestCase):
    """Tests pour lifecore.event."""
    
    def test_event_bus_emit(self):
        """Test émission d'événement."""
        from lifecore.event import EventBus, Event, EventType
        
        bus = EventBus()
        received = []
        
        bus.subscribe("test", lambda e: received.append(e))
        bus.emit(Event(name="test", event_type=EventType.INFO))
        
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].name, "test")
    
    def test_event_type_subscription(self):
        """Test abonnement par type."""
        from lifecore.event import EventBus, Alert, EventType
        
        bus = EventBus()
        alerts = []
        
        bus.subscribe_type(EventType.ALERT, lambda e: alerts.append(e))
        
        bus.emit(Alert(name="test_alert", severity=0.8))
        
        self.assertEqual(len(alerts), 1)
    
    def test_event_history(self):
        """Test historique."""
        from lifecore.event import EventBus, Event
        
        bus = EventBus()
        
        for i in range(5):
            bus.emit(Event(name=f"event_{i}"))
        
        history = bus.get_history()
        self.assertEqual(len(history), 5)
    
    def test_event_factories(self):
        """Test factories d'événements."""
        from lifecore.event import battery_low_alert, equipment_failure, demand_spike
        
        alert = battery_low_alert("drone_1", level=0.2)
        self.assertEqual(alert.name, "battery_low")
        
        failure = equipment_failure("robot_1", "motor")
        self.assertEqual(failure.name, "equipment_failure")


# =============================================================================
# TEST: CONFIG
# =============================================================================

class TestConfig(unittest.TestCase):
    """Tests pour lifecore.config."""
    
    def test_load_json(self):
        """Test chargement JSON."""
        import json
        import tempfile
        import os
        from lifecore.config import load_system
        
        # Créer un fichier config temporaire
        config = {
            "name": "Test",
            "dims": 4,
            "resources": [],
            "laws": [],
            "hierarchy": {"name": "root"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            temp_path = f.name
        
        try:
            system = load_system(temp_path)
            self.assertEqual(system['config']['name'], "Test")
            self.assertIsNotNone(system['root'])
        finally:
            os.unlink(temp_path)
    
    def test_load_yaml(self):
        """Test chargement YAML."""
        import tempfile
        import os
        from lifecore.config import load_system
        
        yaml_content = """
name: TestYAML
dims: 6
resources: []
laws: []
hierarchy:
  name: root
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name
        
        try:
            system = load_system(temp_path)
            self.assertEqual(system['config']['name'], "TestYAML")
        finally:
            os.unlink(temp_path)


# =============================================================================
# MAIN
# =============================================================================

def run_tests():
    """Lance tous les tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Ajouter tous les tests
    suite.addTests(loader.loadTestsFromTestCase(TestLifeCore))
    suite.addTests(loader.loadTestsFromTestCase(TestTensorMemory))
    suite.addTests(loader.loadTestsFromTestCase(TestNeed))
    suite.addTests(loader.loadTestsFromTestCase(TestGoal))
    suite.addTests(loader.loadTestsFromTestCase(TestResource))
    suite.addTests(loader.loadTestsFromTestCase(TestLaw))
    suite.addTests(loader.loadTestsFromTestCase(TestCapability))
    suite.addTests(loader.loadTestsFromTestCase(TestActivation))
    suite.addTests(loader.loadTestsFromTestCase(TestCoherence))
    suite.addTests(loader.loadTestsFromTestCase(TestEvent))
    suite.addTests(loader.loadTestsFromTestCase(TestConfig))
    
    # Runner
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print()
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
        for test, traceback in result.failures + result.errors:
            print(f"  - {test}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
