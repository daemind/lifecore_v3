#!/usr/bin/env python3
"""
LifeCore V3 - Pipeline Module
=============================

Modélisation de pipelines de production paramétrables.
Essentiellement un Gantt dynamique qui s'adapte en temps réel.

Concepts:
- Stage: Étape du pipeline avec durée, ressources, dépendances
- Pipeline: Enchaînement de stages
- Job: Instance d'un pipeline (une commande, un produit)
- Resource: Capacité partagée entre stages

Usage:
    from lifecore.pipeline import Pipeline, Stage, Job
    
    # Définir un pipeline Amazon
    pipe = Pipeline("fulfillment")
    pipe.add_stage(Stage("picking", duration=10, resources={"robot": 1}))
    pipe.add_stage(Stage("transport", duration=5, resources={"robot": 1}))
    pipe.add_stage(Stage("packing", duration=3, resources={"station": 1}))
    pipe.add_stage(Stage("shipping", duration=1, resources={"dock": 1}))
    
    # Dépendances
    pipe.add_dependency("picking", "transport")
    pipe.add_dependency("transport", "packing")
    pipe.add_dependency("packing", "shipping")
    
    # Créer des jobs
    job = pipe.create_job(priority=9, deadline=60)
    
    # Simuler
    pipe.step()  # Avance d'une unité de temps
    
    # Export Gantt
    gantt = pipe.export_gantt()
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json


class JobStatus(Enum):
    """Status d'un job dans le pipeline."""
    PENDING = "pending"         # Pas encore commencé
    WAITING = "waiting"         # Attend une ressource
    IN_PROGRESS = "in_progress" # En cours
    COMPLETED = "completed"     # Terminé
    FAILED = "failed"           # Échec (deadline, erreur)


@dataclass
class Stage:
    """Étape d'un pipeline.
    
    Attributes:
        name: Nom unique de l'étape
        duration: Durée de base (unités de temps)
        resources: Dict {resource_name: quantité requise}
        parallelism: Nombre max de jobs simultanés
        variability: Variation de durée (0-1, ex: 0.2 = ±20%)
    """
    name: str
    duration: float = 1.0
    resources: Dict[str, float] = field(default_factory=dict)
    parallelism: int = 1
    variability: float = 0.0
    
    def actual_duration(self) -> float:
        """Durée avec variabilité."""
        if self.variability > 0:
            factor = 1.0 + np.random.uniform(-self.variability, self.variability)
            return self.duration * factor
        return self.duration


@dataclass
class JobStageState:
    """État d'un job dans une étape."""
    stage_name: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    expected_duration: float = 0.0  # Cached duration
    progress: float = 0.0
    status: JobStatus = JobStatus.PENDING


@dataclass
class Job:
    """Instance d'un pipeline (une commande, un produit).
    
    Attributes:
        id: Identifiant unique
        priority: Priorité (0-10, 10 = plus urgent)
        deadline: Temps limite (None = pas de limite)
        creation_time: Moment de création
        stages: État dans chaque stage
        metadata: Données additionnelles
    """
    id: int
    priority: float = 5.0
    deadline: Optional[float] = None
    creation_time: float = 0.0
    current_stage: Optional[str] = None
    stages: Dict[str, JobStageState] = field(default_factory=dict)
    metadata: Dict[str, any] = field(default_factory=dict)
    completion_time: Optional[float] = None
    
    @property
    def status(self) -> JobStatus:
        """Status global du job."""
        if self.completion_time is not None:
            return JobStatus.COMPLETED
        if any(s.status == JobStatus.FAILED for s in self.stages.values()):
            return JobStatus.FAILED
        if any(s.status == JobStatus.IN_PROGRESS for s in self.stages.values()):
            return JobStatus.IN_PROGRESS
        if any(s.status == JobStatus.WAITING for s in self.stages.values()):
            return JobStatus.WAITING
        return JobStatus.PENDING
    
    def time_in_pipeline(self, current_time: float) -> float:
        """Temps passé dans le pipeline."""
        return current_time - self.creation_time


@dataclass
class ResourcePool:
    """Pool de ressources partagées.
    
    Attributes:
        name: Nom de la ressource
        capacity: Capacité totale
        allocated: Quantité actuellement allouée
    """
    name: str
    capacity: float = 1.0
    allocated: float = 0.0
    
    @property
    def available(self) -> float:
        return self.capacity - self.allocated
    
    def allocate(self, amount: float) -> bool:
        """Alloue des ressources. Retourne True si succès."""
        if amount <= self.available:
            self.allocated += amount
            return True
        return False
    
    def release(self, amount: float) -> None:
        """Libère des ressources."""
        self.allocated = max(0, self.allocated - amount)


class Pipeline:
    """Pipeline de production paramétrable.
    
    Un pipeline est un DAG (directed acyclic graph) de stages.
    Chaque stage consomme des ressources pendant une durée.
    Les jobs traversent le pipeline selon les dépendances.
    
    Example:
        pipe = Pipeline("manufacturing")
        pipe.add_stage(Stage("cut", duration=5, resources={"machine": 1}))
        pipe.add_stage(Stage("assemble", duration=10, resources={"worker": 2}))
        pipe.add_stage(Stage("test", duration=3, resources={"tester": 1}))
        
        pipe.add_dependency("cut", "assemble")
        pipe.add_dependency("assemble", "test")
        
        pipe.set_resource("machine", capacity=5)
        pipe.set_resource("worker", capacity=20)
        pipe.set_resource("tester", capacity=3)
    """
    
    def __init__(self, name: str):
        self.name = name
        self.stages: Dict[str, Stage] = {}
        self.dependencies: Dict[str, Set[str]] = defaultdict(set)  # stage -> set of prerequisites
        self.resources: Dict[str, ResourcePool] = {}
        self.jobs: Dict[int, Job] = {}
        self.completed_jobs: List[Job] = []
        self.failed_jobs: List[Job] = []
        
        self.current_time: float = 0.0
        self._next_job_id: int = 0
        
        # Stats
        self.stats = {
            "jobs_created": 0,
            "jobs_completed": 0,
            "jobs_failed": 0,
            "total_time_in_system": 0.0,
            "utilization": defaultdict(float)
        }
    
    # === CONFIGURATION ===
    
    def add_stage(self, stage: Stage) -> None:
        """Ajoute une étape au pipeline."""
        self.stages[stage.name] = stage
    
    def add_dependency(self, prerequisite: str, dependent: str) -> None:
        """Ajoute une dépendance: 'dependent' nécessite 'prerequisite'."""
        if prerequisite not in self.stages:
            raise ValueError(f"Stage '{prerequisite}' not found")
        if dependent not in self.stages:
            raise ValueError(f"Stage '{dependent}' not found")
        self.dependencies[dependent].add(prerequisite)
    
    def set_resource(self, name: str, capacity: float) -> None:
        """Définit une ressource."""
        self.resources[name] = ResourcePool(name=name, capacity=capacity)
    
    def get_entry_stages(self) -> List[str]:
        """Retourne les stages d'entrée (sans prérequis)."""
        return [name for name in self.stages if not self.dependencies.get(name)]
    
    def get_exit_stages(self) -> List[str]:
        """Retourne les stages de sortie (personne ne dépend d'eux)."""
        dependents = set()
        for deps in self.dependencies.values():
            dependents.update(deps)
        return [name for name in self.stages if name not in dependents or 
                not any(name in self.dependencies.get(s, set()) for s in self.stages)]
    
    def get_next_stages(self, stage_name: str) -> List[str]:
        """Retourne les stages qui suivent (qui dépendent de ce stage)."""
        return [name for name, deps in self.dependencies.items() if stage_name in deps]
    
    # === JOBS ===
    
    def create_job(self, priority: float = 5.0, deadline: Optional[float] = None,
                   **metadata) -> Job:
        """Crée un nouveau job."""
        job = Job(
            id=self._next_job_id,
            priority=priority,
            deadline=deadline,
            creation_time=self.current_time,
            metadata=metadata
        )
        
        # Initialiser les états de stages
        for stage_name in self.stages:
            job.stages[stage_name] = JobStageState(stage_name=stage_name)
        
        self.jobs[job.id] = job
        self._next_job_id += 1
        self.stats["jobs_created"] += 1
        
        return job
    
    def _can_start_stage(self, job: Job, stage_name: str) -> bool:
        """Vérifie si un job peut commencer un stage."""
        # Vérifier les prérequis
        for prereq in self.dependencies.get(stage_name, set()):
            prereq_state = job.stages.get(prereq)
            if not prereq_state or prereq_state.status != JobStatus.COMPLETED:
                return False
        
        # Vérifier les ressources
        stage = self.stages[stage_name]
        for res_name, amount in stage.resources.items():
            pool = self.resources.get(res_name)
            if not pool or pool.available < amount:
                return False
        
        return True
    
    def _start_stage(self, job: Job, stage_name: str) -> None:
        """Démarre un job sur un stage."""
        stage = self.stages[stage_name]
        state = job.stages[stage_name]
        
        # Allouer les ressources
        for res_name, amount in stage.resources.items():
            self.resources[res_name].allocate(amount)
        
        state.start_time = self.current_time
        state.expected_duration = stage.actual_duration()  # Cache duration
        state.status = JobStatus.IN_PROGRESS
        state.progress = 0.0
        job.current_stage = stage_name
    
    def _complete_stage(self, job: Job, stage_name: str) -> None:
        """Termine un job sur un stage."""
        stage = self.stages[stage_name]
        state = job.stages[stage_name]
        
        # Libérer les ressources
        for res_name, amount in stage.resources.items():
            self.resources[res_name].release(amount)
        
        state.end_time = self.current_time
        state.status = JobStatus.COMPLETED
        state.progress = 1.0
        
        # Vérifier si le job est terminé
        next_stages = self.get_next_stages(stage_name)
        if not next_stages:
            job.completion_time = self.current_time
            self.completed_jobs.append(job)
            del self.jobs[job.id]
            self.stats["jobs_completed"] += 1
            self.stats["total_time_in_system"] += job.time_in_pipeline(self.current_time)
    
    def _fail_job(self, job: Job, reason: str = "deadline") -> None:
        """Marque un job comme échoué."""
        for state in job.stages.values():
            if state.status == JobStatus.IN_PROGRESS:
                # Libérer les ressources
                stage = self.stages[state.stage_name]
                for res_name, amount in stage.resources.items():
                    self.resources[res_name].release(amount)
                state.status = JobStatus.FAILED
        
        self.failed_jobs.append(job)
        del self.jobs[job.id]
        self.stats["jobs_failed"] += 1
    
    # === SIMULATION ===
    
    def step(self, dt: float = 1.0) -> None:
        """Avance la simulation de dt unités de temps."""
        self.current_time += dt
        
        # Trier les jobs par priorité - copie pour éviter modification pendant itération
        sorted_jobs = sorted(list(self.jobs.values()), key=lambda j: -j.priority)
        
        for job in sorted_jobs:
            if job.id not in self.jobs:  # Job déjà retiré
                continue
                
            # Vérifier deadline
            if job.deadline and job.time_in_pipeline(self.current_time) > job.deadline:
                self._fail_job(job, "deadline")
                continue
            
            # Mettre à jour les stages en cours
            for stage_name, state in job.stages.items():
                if state.status == JobStatus.IN_PROGRESS:
                    elapsed = self.current_time - state.start_time
                    state.progress = min(1.0, elapsed / max(0.1, state.expected_duration))
                    
                    if state.progress >= 1.0:
                        self._complete_stage(job, stage_name)
            
            if job.id not in self.jobs:  # Job terminé
                continue
            
            # Essayer de démarrer de nouveaux stages
            for stage_name in self.stages:
                state = job.stages[stage_name]
                # Check both PENDING and WAITING (waiting = tried before, resources unavailable)
                if state.status in (JobStatus.PENDING, JobStatus.WAITING):
                    if self._can_start_stage(job, stage_name):
                        self._start_stage(job, stage_name)
                        break  # Un seul stage à la fois par job
                    elif state.status == JobStatus.PENDING:
                        state.status = JobStatus.WAITING
        
        # Calculer l'utilisation
        for res_name, pool in self.resources.items():
            utilization = pool.allocated / pool.capacity if pool.capacity > 0 else 0
            self.stats["utilization"][res_name] = (
                self.stats["utilization"][res_name] * 0.99 + utilization * 0.01
            )
    
    def run(self, duration: float, dt: float = 1.0) -> Dict:
        """Lance la simulation pour une durée donnée."""
        steps = int(duration / dt)
        for _ in range(steps):
            self.step(dt)
        return self.get_stats()
    
    # === STATS & EXPORT ===
    
    def get_stats(self) -> Dict:
        """Retourne les statistiques du pipeline."""
        avg_time = (self.stats["total_time_in_system"] / 
                   max(1, self.stats["jobs_completed"]))
        
        return {
            "name": self.name,
            "current_time": self.current_time,
            "jobs_created": self.stats["jobs_created"],
            "jobs_completed": self.stats["jobs_completed"],
            "jobs_failed": self.stats["jobs_failed"],
            "jobs_in_progress": len(self.jobs),
            "avg_time_in_system": avg_time,
            "completion_rate": (self.stats["jobs_completed"] / 
                               max(1, self.stats["jobs_created"])),
            "resource_utilization": dict(self.stats["utilization"])
        }
    
    def export_gantt(self, max_jobs: int = 50) -> List[Dict]:
        """Exporte les données au format Gantt.
        
        Returns:
            Liste de dict: {job_id, stage, start, end, status}
        """
        gantt_data = []
        
        # Jobs complétés
        for job in self.completed_jobs[-max_jobs:]:
            for stage_name, state in job.stages.items():
                if state.start_time is not None:
                    gantt_data.append({
                        "job_id": job.id,
                        "stage": stage_name,
                        "start": state.start_time,
                        "end": state.end_time or self.current_time,
                        "status": state.status.value
                    })
        
        # Jobs en cours
        for job in self.jobs.values():
            for stage_name, state in job.stages.items():
                if state.start_time is not None:
                    gantt_data.append({
                        "job_id": job.id,
                        "stage": stage_name,
                        "start": state.start_time,
                        "end": state.end_time or self.current_time,
                        "status": state.status.value
                    })
        
        return sorted(gantt_data, key=lambda x: (x["start"], x["job_id"]))
    
    def export_json(self) -> str:
        """Exporte le pipeline en JSON."""
        return json.dumps({
            "name": self.name,
            "stages": [
                {
                    "name": s.name,
                    "duration": s.duration,
                    "resources": s.resources,
                    "parallelism": s.parallelism
                }
                for s in self.stages.values()
            ],
            "dependencies": {
                stage: list(deps) 
                for stage, deps in self.dependencies.items()
            },
            "resources": {
                name: pool.capacity 
                for name, pool in self.resources.items()
            },
            "stats": self.get_stats()
        }, indent=2)


# === PRESET PIPELINES ===

def create_fulfillment_pipeline(
    robots: int = 50,
    packing_stations: int = 10,
    shipping_docks: int = 5
) -> Pipeline:
    """Crée un pipeline de fulfillment Amazon-like."""
    pipe = Pipeline("fulfillment")
    
    # Stages - durées courtes pour demo
    pipe.add_stage(Stage("picking", duration=5, resources={"robot": 1}, variability=0.2))
    pipe.add_stage(Stage("transport", duration=3, resources={"robot": 1}))
    pipe.add_stage(Stage("packing", duration=2, resources={"station": 1}, variability=0.3))
    pipe.add_stage(Stage("shipping", duration=1, resources={"dock": 1}))
    
    # Dépendances
    pipe.add_dependency("picking", "transport")
    pipe.add_dependency("transport", "packing")
    pipe.add_dependency("packing", "shipping")
    
    # Ressources
    pipe.set_resource("robot", capacity=robots)
    pipe.set_resource("station", capacity=packing_stations)
    pipe.set_resource("dock", capacity=shipping_docks)
    
    return pipe


def create_manufacturing_pipeline(
    machines: int = 10,
    workers: int = 20,
    testers: int = 3
) -> Pipeline:
    """Crée un pipeline de fabrication."""
    pipe = Pipeline("manufacturing")
    
    # Stages
    pipe.add_stage(Stage("raw_material", duration=2, resources={}))
    pipe.add_stage(Stage("cutting", duration=5, resources={"machine": 1}))
    pipe.add_stage(Stage("assembly", duration=15, resources={"worker": 2}, variability=0.2))
    pipe.add_stage(Stage("quality_check", duration=3, resources={"tester": 1}))
    pipe.add_stage(Stage("packaging", duration=2, resources={"worker": 1}))
    
    # Dépendances
    pipe.add_dependency("raw_material", "cutting")
    pipe.add_dependency("cutting", "assembly")
    pipe.add_dependency("assembly", "quality_check")
    pipe.add_dependency("quality_check", "packaging")
    
    # Ressources
    pipe.set_resource("machine", capacity=machines)
    pipe.set_resource("worker", capacity=workers)
    pipe.set_resource("tester", capacity=testers)
    
    return pipe


# === DEMO ===

if __name__ == "__main__":
    print("Pipeline Module Demo")
    print("=" * 50)
    
    # Créer un pipeline - plus de ressources
    pipe = create_fulfillment_pipeline(robots=50, packing_stations=20, shipping_docks=10)
    print(f"Pipeline: {pipe.name}")
    print(f"Stages: {list(pipe.stages.keys())}")
    print(f"Resources: {list(pipe.resources.keys())}")
    print()
    
    # Créer des jobs - deadlines réalistes
    for i in range(30):
        priority = np.random.choice([9, 5, 2], p=[0.3, 0.5, 0.2])
        deadline = {9: 50, 5: 100, 2: 200}[priority]  # Plus longs
        pipe.create_job(priority=priority, deadline=deadline)
    
    print(f"Jobs créés: {pipe.stats['jobs_created']}")
    print(f"Stages duration: picking=10, transport=5, packing=3, shipping=1 = 19 min")
    print()
    
    # Simuler
    print("Simulation 80 steps...")
    for step in range(80):
        pipe.step()
        if (step + 1) % 20 == 0:
            stats = pipe.get_stats()
            util = stats['resource_utilization']
            print(f"  Step {step+1}: {stats['jobs_completed']} completed, "
                  f"{stats['jobs_failed']} failed, {stats['jobs_in_progress']} in progress")
            print(f"    Utilization: robot={util.get('robot', 0):.1%}, "
                  f"station={util.get('station', 0):.1%}, dock={util.get('dock', 0):.1%}")
    
    print()
    print("Résultats finaux:")
    stats = pipe.get_stats()
    print(f"  Completed: {stats['jobs_completed']}/{stats['jobs_created']}")
    print(f"  Failed: {stats['jobs_failed']}")
    print(f"  Avg time: {stats['avg_time_in_system']:.1f}")
    
    # Export Gantt
    gantt = pipe.export_gantt(max_jobs=10)
    print()
    print(f"Gantt ({len(gantt)} entries, first 10):")
    for entry in gantt[:10]:
        print(f"  Job {entry['job_id']}: {entry['stage']} "
              f"[{entry['start']:.0f}-{entry['end']:.0f}] {entry['status']}")
