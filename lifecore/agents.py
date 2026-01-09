#!/usr/bin/env python3
"""
LifeCore V3 - Specialized Agents
================================

Agents spécialisés pour différents rôles dans une chaîne de production.
Chaque type d'agent hérite de LifeCore et ajoute des comportements spécifiques.

Types d'agents:
- MachineAgent: Machine outil (CNC, robot soudeur, etc.)
- TransportAgent: AGV, convoyeur, chariot
- MaintenanceAgent: Technicien de maintenance
- QualityAgent: Contrôle qualité
- StorageAgent: Stockage/Entrepôt
- OperatorAgent: Opérateur humain

Usage:
    from lifecore.agents import MachineAgent, MaintenanceAgent
    
    cnc = MachineAgent(name="CNC_01", capacity=10, cycle_time=5)
    tech = MaintenanceAgent(name="Tech_01", skills=["mechanical", "electrical"])
    
    # Les agents interagissent via le pipeline
    pipeline.add_worker(cnc, stage="machining")
    pipeline.add_worker(tech, stage="maintenance")
"""

import numpy as np
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

# Import core
try:
    from .core import LifeCore
    from .pipeline import Pipeline, Stage, Job
    from .event import EventBus, Event, EventType
except ImportError:
    from core import LifeCore
    from pipeline import Pipeline, Stage, Job
    from event import EventBus, Event, EventType


# === ENUMS ===

class AgentStatus(Enum):
    """Status d'un agent."""
    IDLE = "idle"
    WORKING = "working"
    MAINTENANCE = "maintenance"
    FAILURE = "failure"
    CHARGING = "charging"
    OFFLINE = "offline"


class MaintenanceType(Enum):
    """Type de maintenance."""
    PREVENTIVE = "preventive"
    CORRECTIVE = "corrective"
    PREDICTIVE = "predictive"


# === BASE AGENT ===

@dataclass
class AgentState:
    """État générique d'un agent."""
    health: float = 1.0           # 0-1, santé de l'équipement
    energy: float = 1.0           # 0-1, batterie/énergie
    utilization: float = 0.0      # 0-1, taux d'utilisation
    cycles_count: int = 0         # Nombre de cycles effectués
    errors_count: int = 0         # Nombre d'erreurs
    last_maintenance: float = 0.0 # Temps depuis dernière maintenance


class BaseAgent(ABC):
    """Agent de base pour la chaîne de production."""
    
    def __init__(
        self,
        name: str,
        agent_type: str,
        position: tuple = (0, 0),
        event_bus: Optional[EventBus] = None
    ):
        self.name = name
        self.agent_type = agent_type
        self.position = position
        self.event_bus = event_bus or EventBus()
        
        self.status = AgentStatus.IDLE
        self.state = AgentState()
        self.current_job: Optional[Job] = None
        self.history: List[Dict] = []
        
        # Capabilities
        self.capabilities: Set[str] = set()
        
        # Internal LifeCore for decision making
        self.core = LifeCore(dims=4)
    
    @abstractmethod
    def step(self, dt: float = 1.0) -> None:
        """Avance l'agent d'un pas de temps."""
        pass
    
    @abstractmethod
    def can_handle(self, job: Job) -> bool:
        """Vérifie si l'agent peut traiter ce job."""
        pass
    
    def assign_job(self, job: Job) -> bool:
        """Assigne un job à l'agent."""
        if self.status != AgentStatus.IDLE:
            return False
        if not self.can_handle(job):
            return False
        
        self.current_job = job
        self.status = AgentStatus.WORKING
        self._log("job_assigned", {"job_id": job.id})
        return True
    
    def complete_job(self) -> Optional[Job]:
        """Termine le job actuel."""
        if not self.current_job:
            return None
        
        job = self.current_job
        self.current_job = None
        self.status = AgentStatus.IDLE
        self.state.cycles_count += 1
        
        self._log("job_completed", {"job_id": job.id})
        return job
    
    def trigger_failure(self, reason: str = "unknown") -> None:
        """Déclenche une panne."""
        self.status = AgentStatus.FAILURE
        self.state.errors_count += 1
        self.state.health *= 0.8  # Dégradation
        
        self._log("failure", {"reason": reason})
        
        if self.event_bus:
            self.event_bus.emit(Event(
                source=self.name,
                event_type=EventType.FAILURE,
                severity=0.8,
                data={"reason": reason, "agent": self.name}
            ))
    
    def request_maintenance(self, mtype: MaintenanceType = MaintenanceType.CORRECTIVE) -> None:
        """Demande une maintenance."""
        self._log("maintenance_requested", {"type": mtype.value})
        
        if self.event_bus:
            self.event_bus.emit(Event(
                source=self.name,
                event_type=EventType.DEMAND,
                severity=0.6,
                data={"type": mtype.value, "agent": self.name}
            ))
    
    def perform_maintenance(self, duration: float = 10.0) -> None:
        """Effectue la maintenance."""
        self.status = AgentStatus.MAINTENANCE
        self.state.health = min(1.0, self.state.health + 0.3)
        self.state.last_maintenance = 0.0
        
        self._log("maintenance_performed", {"health": self.state.health})
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques de l'agent."""
        return {
            "name": self.name,
            "type": self.agent_type,
            "status": self.status.value,
            "health": self.state.health,
            "energy": self.state.energy,
            "utilization": self.state.utilization,
            "cycles": self.state.cycles_count,
            "errors": self.state.errors_count
        }
    
    def _log(self, action: str, data: Dict) -> None:
        """Log une action."""
        self.history.append({
            "action": action,
            "data": data,
            "status": self.status.value
        })


# === MACHINE AGENT ===

class MachineAgent(BaseAgent):
    """Agent machine outil (CNC, robot soudeur, presse, etc.)
    
    Caractéristiques:
    - Cycle time fixe ou variable
    - Capacité (pièces/heure)
    - Dégradation avec l'usage
    - Probabilité de panne
    """
    
    def __init__(
        self,
        name: str,
        cycle_time: float = 5.0,
        capacity: int = 10,
        failure_rate: float = 0.001,
        **kwargs
    ):
        super().__init__(name, "machine", **kwargs)
        self.cycle_time = cycle_time
        self.capacity = capacity
        self.failure_rate = failure_rate
        
        self.current_cycle_progress = 0.0
        self.parts_produced = 0
        
        self.capabilities.add("machining")
    
    def step(self, dt: float = 1.0) -> None:
        """Avance la machine d'un pas."""
        self.state.last_maintenance += dt
        
        if self.status == AgentStatus.WORKING:
            self.current_cycle_progress += dt
            self.state.utilization = 0.9 * self.state.utilization + 0.1
            
            # Vérifier fin de cycle
            if self.current_cycle_progress >= self.cycle_time:
                self.current_cycle_progress = 0.0
                self.parts_produced += 1
                self.complete_job()
            
            # Probabilité de panne (augmente avec l'usure)
            failure_prob = self.failure_rate * (2 - self.state.health)
            if np.random.random() < failure_prob:
                self.trigger_failure("random_failure")
        
        elif self.status == AgentStatus.IDLE:
            self.state.utilization = 0.9 * self.state.utilization
        
        # Dégradation naturelle
        self.state.health -= 0.0001 * dt
        
        # Vérifier besoin maintenance préventive
        if self.state.health < 0.3:
            self.request_maintenance(MaintenanceType.PREVENTIVE)
    
    def can_handle(self, job: Job) -> bool:
        return self.status == AgentStatus.IDLE and self.state.health > 0.2


# === TRANSPORT AGENT ===

class TransportAgent(BaseAgent):
    """Agent de transport (AGV, chariot, convoyeur).
    
    Caractéristiques:
    - Vitesse de déplacement
    - Capacité de charge
    - Navigation
    - Batterie
    """
    
    def __init__(
        self,
        name: str,
        speed: float = 2.0,
        max_load: float = 100.0,
        battery_capacity: float = 100.0,
        **kwargs
    ):
        super().__init__(name, "transport", **kwargs)
        self.speed = speed
        self.max_load = max_load
        self.battery_capacity = battery_capacity
        
        self.target_position: Optional[tuple] = None
        self.current_load: float = 0.0
        self.total_distance: float = 0.0
        
        self.capabilities.add("transport")
        self.capabilities.add("pickup")
        self.capabilities.add("delivery")
    
    def step(self, dt: float = 1.0) -> None:
        """Avance l'AGV d'un pas."""
        if self.status == AgentStatus.WORKING and self.target_position:
            # Calculer la direction
            dx = self.target_position[0] - self.position[0]
            dy = self.target_position[1] - self.position[1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist < 0.5:
                # Arrivé
                self.position = self.target_position
                self.target_position = None
                self.complete_job()
            else:
                # Avancer
                move = min(self.speed * dt, dist)
                self.position = (
                    self.position[0] + (dx / dist) * move,
                    self.position[1] + (dy / dist) * move
                )
                self.total_distance += move
                
                # Consommer batterie
                drain = 0.01 * move * (1 + self.current_load / self.max_load)
                self.state.energy -= drain
        
        elif self.status == AgentStatus.CHARGING:
            self.state.energy = min(1.0, self.state.energy + 0.05 * dt)
            if self.state.energy >= 0.95:
                self.status = AgentStatus.IDLE
        
        # Vérifier batterie faible
        if self.state.energy < 0.2 and self.status != AgentStatus.CHARGING:
            self.status = AgentStatus.CHARGING
            self._log("low_battery", {"energy": self.state.energy})
    
    def can_handle(self, job: Job) -> bool:
        return (self.status == AgentStatus.IDLE and 
                self.state.energy > 0.3)
    
    def set_destination(self, position: tuple) -> None:
        """Définit la destination."""
        self.target_position = position


# === MAINTENANCE AGENT ===

class MaintenanceAgent(BaseAgent):
    """Agent de maintenance (technicien).
    
    Caractéristiques:
    - Skills (mécanique, électrique, etc.)
    - Temps de réparation
    - Efficacité
    """
    
    def __init__(
        self,
        name: str,
        skills: List[str] = None,
        repair_speed: float = 1.0,
        **kwargs
    ):
        super().__init__(name, "maintenance", **kwargs)
        self.skills = set(skills or ["mechanical", "electrical"])
        self.repair_speed = repair_speed
        
        self.current_repair_target: Optional[BaseAgent] = None
        self.repair_progress: float = 0.0
        self.repairs_completed: int = 0
        
        self.capabilities.add("repair")
        self.capabilities.add("maintenance")
    
    def step(self, dt: float = 1.0) -> None:
        """Avance le technicien d'un pas."""
        if self.status == AgentStatus.WORKING and self.current_repair_target:
            self.repair_progress += dt * self.repair_speed
            
            # Réparer progressivement
            if self.repair_progress >= 10.0:  # 10 unités pour réparer
                self.current_repair_target.state.health = min(1.0, 
                    self.current_repair_target.state.health + 0.5)
                self.current_repair_target.status = AgentStatus.IDLE
                self.current_repair_target.state.last_maintenance = 0.0
                
                self.repairs_completed += 1
                self.repair_progress = 0.0
                self.current_repair_target = None
                self.status = AgentStatus.IDLE
                
                self._log("repair_completed", {"repairs": self.repairs_completed})
    
    def can_handle(self, job: Job) -> bool:
        return self.status == AgentStatus.IDLE
    
    def assign_repair(self, target: BaseAgent) -> bool:
        """Assigne une réparation."""
        if self.status != AgentStatus.IDLE:
            return False
        
        self.current_repair_target = target
        self.status = AgentStatus.WORKING
        self.repair_progress = 0.0
        
        self._log("repair_assigned", {"target": target.name})
        return True


# === QUALITY AGENT ===

class QualityAgent(BaseAgent):
    """Agent de contrôle qualité.
    
    Caractéristiques:
    - Précision de détection
    - Temps d'inspection
    - Critères de qualité
    """
    
    def __init__(
        self,
        name: str,
        inspection_time: float = 2.0,
        accuracy: float = 0.95,
        **kwargs
    ):
        super().__init__(name, "quality", **kwargs)
        self.inspection_time = inspection_time
        self.accuracy = accuracy
        
        self.inspection_progress: float = 0.0
        self.items_inspected: int = 0
        self.defects_found: int = 0
        
        self.capabilities.add("inspection")
        self.capabilities.add("quality_control")
    
    def step(self, dt: float = 1.0) -> None:
        """Avance l'inspection d'un pas."""
        if self.status == AgentStatus.WORKING:
            self.inspection_progress += dt
            
            if self.inspection_progress >= self.inspection_time:
                # Terminer l'inspection
                self.items_inspected += 1
                
                # Simuler détection de défaut
                has_defect = np.random.random() < 0.05  # 5% de défauts
                detected = np.random.random() < self.accuracy
                
                if has_defect and detected:
                    self.defects_found += 1
                    self._log("defect_detected", {"item": self.items_inspected})
                
                self.inspection_progress = 0.0
                self.complete_job()
    
    def can_handle(self, job: Job) -> bool:
        return self.status == AgentStatus.IDLE


# === STORAGE AGENT ===

class StorageAgent(BaseAgent):
    """Agent de stockage (zone de stockage, buffer).
    
    Caractéristiques:
    - Capacité max
    - FIFO/LIFO
    - Temps de picking
    """
    
    def __init__(
        self,
        name: str,
        capacity: int = 100,
        picking_time: float = 1.0,
        **kwargs
    ):
        super().__init__(name, "storage", **kwargs)
        self.capacity = capacity
        self.picking_time = picking_time
        
        self.inventory: List[Any] = []
        self.picks_completed: int = 0
        
        self.capabilities.add("storage")
        self.capabilities.add("picking")
    
    def step(self, dt: float = 1.0) -> None:
        """Pas de simulation."""
        self.state.utilization = len(self.inventory) / self.capacity
    
    def can_handle(self, job: Job) -> bool:
        return len(self.inventory) < self.capacity
    
    def store(self, item: Any) -> bool:
        """Stocke un item."""
        if len(self.inventory) >= self.capacity:
            return False
        self.inventory.append(item)
        self._log("stored", {"count": len(self.inventory)})
        return True
    
    def pick(self) -> Optional[Any]:
        """Récupère un item (FIFO)."""
        if not self.inventory:
            return None
        item = self.inventory.pop(0)
        self.picks_completed += 1
        return item


# === FACTORY ===

class Factory:
    """Usine complète avec tous les agents.
    
    Coordonne les différents types d'agents.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.agents: Dict[str, BaseAgent] = {}
        self.event_bus = EventBus()
        self.current_time: float = 0.0
        
        # Stats
        self.stats = {
            "parts_produced": 0,
            "defects_found": 0,
            "failures": 0,
            "repairs": 0
        }
    
    def add_agent(self, agent: BaseAgent) -> None:
        """Ajoute un agent à l'usine."""
        agent.event_bus = self.event_bus
        self.agents[agent.name] = agent
    
    def add_machine(self, name: str, **kwargs) -> MachineAgent:
        """Ajoute une machine."""
        agent = MachineAgent(name, event_bus=self.event_bus, **kwargs)
        self.agents[name] = agent
        return agent
    
    def add_transport(self, name: str, **kwargs) -> TransportAgent:
        """Ajoute un AGV."""
        agent = TransportAgent(name, event_bus=self.event_bus, **kwargs)
        self.agents[name] = agent
        return agent
    
    def add_maintenance_tech(self, name: str, **kwargs) -> MaintenanceAgent:
        """Ajoute un technicien."""
        agent = MaintenanceAgent(name, event_bus=self.event_bus, **kwargs)
        self.agents[name] = agent
        return agent
    
    def add_quality_station(self, name: str, **kwargs) -> QualityAgent:
        """Ajoute un poste QC."""
        agent = QualityAgent(name, event_bus=self.event_bus, **kwargs)
        self.agents[name] = agent
        return agent
    
    def step(self, dt: float = 1.0) -> None:
        """Avance l'usine d'un pas."""
        self.current_time += dt
        
        # Mettre à jour tous les agents
        for agent in self.agents.values():
            agent.step(dt)
        
        # Assigner les techniciens aux pannes
        self._assign_repairs()
    
    def _assign_repairs(self) -> None:
        """Assigne les techniciens aux machines en panne."""
        techs = [a for a in self.agents.values() 
                 if isinstance(a, MaintenanceAgent) and a.status == AgentStatus.IDLE]
        
        failures = [a for a in self.agents.values() 
                   if a.status == AgentStatus.FAILURE]
        
        for failure, tech in zip(failures, techs):
            tech.assign_repair(failure)
            self.stats["repairs"] += 1
    
    def get_status(self) -> Dict:
        """Retourne le status de l'usine."""
        by_type = {}
        for agent in self.agents.values():
            atype = agent.agent_type
            if atype not in by_type:
                by_type[atype] = {"count": 0, "working": 0, "idle": 0, "failure": 0}
            by_type[atype]["count"] += 1
            if agent.status == AgentStatus.WORKING:
                by_type[atype]["working"] += 1
            elif agent.status == AgentStatus.IDLE:
                by_type[atype]["idle"] += 1
            elif agent.status == AgentStatus.FAILURE:
                by_type[atype]["failure"] += 1
        
        return {
            "name": self.name,
            "time": self.current_time,
            "agents": len(self.agents),
            "by_type": by_type,
            "stats": self.stats
        }


# === DEMO ===

if __name__ == "__main__":
    print("Specialized Agents Demo")
    print("=" * 50)
    
    # Créer une usine
    factory = Factory("Usine_Demo")
    
    # Ajouter des machines
    for i in range(5):
        factory.add_machine(f"CNC_{i}", cycle_time=3, capacity=20)
    
    # Ajouter des AGV
    for i in range(3):
        factory.add_transport(f"AGV_{i}", speed=2.0)
    
    # Ajouter techniciens
    for i in range(2):
        factory.add_maintenance_tech(f"Tech_{i}", skills=["mechanical", "electrical"])
    
    # Ajouter QC
    factory.add_quality_station("QC_1", accuracy=0.98)
    
    print(f"Agents: {len(factory.agents)}")
    print()
    
    # Simuler
    print("Simulation 100 steps...")
    for step in range(100):
        # Assigner des jobs aux machines idle
        for name, agent in factory.agents.items():
            if isinstance(agent, MachineAgent) and agent.status == AgentStatus.IDLE:
                # Créer un fake job
                from dataclasses import dataclass
                @dataclass
                class FakeJob:
                    id: int = step
                agent.assign_job(FakeJob())
        
        factory.step()
        
        if (step + 1) % 25 == 0:
            status = factory.get_status()
            print(f"  Step {step+1}: {status['by_type']}")
    
    print()
    print("Final status:")
    status = factory.get_status()
    for atype, data in status["by_type"].items():
        print(f"  {atype}: {data}")
    
    # Stats machines
    print()
    print("Machine stats:")
    for name, agent in factory.agents.items():
        if isinstance(agent, MachineAgent):
            print(f"  {name}: {agent.parts_produced} parts, health={agent.state.health:.2f}")
