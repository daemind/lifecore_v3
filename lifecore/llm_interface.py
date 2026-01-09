#!/usr/bin/env python3
"""
LifeCore V3 - LLM Interface
============================

Interface LLM pour contrôler LifeCore en langage naturel.
Permet de configurer, monitorer et contrôler les pipelines via conversation.

Features:
- Configuration en langage naturel
- Monitoring conversationnel
- Explications des bottlenecks
- Simulation de scénarios
- Recommandations automatiques

Usage:
    python lifecore/llm_interface.py
    # Ou avec Gradio:
    python examples/llm_control_tower.py
"""

import json
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

# Import pipeline
try:
    from .pipeline import (
        Pipeline, Stage, Job, JobStatus,
        create_fulfillment_pipeline, create_manufacturing_pipeline
    )
except ImportError:
    from pipeline import (
        Pipeline, Stage, Job, JobStatus,
        create_fulfillment_pipeline, create_manufacturing_pipeline
    )


# === TOOL DEFINITIONS ===

@dataclass
class ToolResult:
    """Résultat d'un appel de tool."""
    success: bool
    message: str
    data: Optional[Dict] = None


class PipelineController:
    """Contrôleur de pipeline exposant des tools pour LLM."""
    
    def __init__(self):
        self.pipeline: Optional[Pipeline] = None
        self.history: List[Dict] = []
    
    # === TOOLS FOR LLM ===
    
    def create_pipeline(
        self,
        pipeline_type: str = "fulfillment",
        robots: int = 50,
        stations: int = 20,
        docks: int = 10,
        custom_stages: Optional[List[Dict]] = None
    ) -> ToolResult:
        """Crée un nouveau pipeline.
        
        Args:
            pipeline_type: Type de pipeline ('fulfillment' ou 'manufacturing')
            robots: Nombre de robots/machines
            stations: Nombre de stations/workers
            docks: Nombre de docks/testers
            custom_stages: Stages personnalisés [{name, duration, resources}]
        
        Returns:
            ToolResult avec le status et les détails du pipeline créé
        """
        try:
            if custom_stages:
                self.pipeline = Pipeline("custom")
                prev_stage = None
                for stage_def in custom_stages:
                    stage = Stage(
                        name=stage_def["name"],
                        duration=stage_def.get("duration", 5),
                        resources=stage_def.get("resources", {}),
                        variability=stage_def.get("variability", 0.1)
                    )
                    self.pipeline.add_stage(stage)
                    if prev_stage:
                        self.pipeline.add_dependency(prev_stage, stage.name)
                    prev_stage = stage.name
                    
                    # Ajouter les ressources
                    for res_name, amount in stage.resources.items():
                        if res_name not in self.pipeline.resources:
                            self.pipeline.set_resource(res_name, capacity=50)
            elif pipeline_type == "fulfillment":
                self.pipeline = create_fulfillment_pipeline(robots, stations, docks)
            else:
                self.pipeline = create_manufacturing_pipeline(robots, stations, docks)
            
            self._log("create_pipeline", {
                "type": pipeline_type,
                "robots": robots,
                "stations": stations
            })
            
            return ToolResult(
                success=True,
                message=f"Pipeline '{self.pipeline.name}' créé avec {len(self.pipeline.stages)} stages",
                data={
                    "name": self.pipeline.name,
                    "stages": list(self.pipeline.stages.keys()),
                    "resources": {k: v.capacity for k, v in self.pipeline.resources.items()}
                }
            )
        except Exception as e:
            return ToolResult(success=False, message=f"Erreur: {str(e)}")
    
    def add_jobs(
        self,
        count: int = 10,
        priority: int = 5,
        deadline: int = 60,
        priority_distribution: Optional[Dict[str, float]] = None
    ) -> ToolResult:
        """Ajoute des jobs au pipeline.
        
        Args:
            count: Nombre de jobs à ajouter
            priority: Priorité (1-10, 10=urgent)
            deadline: Deadline en minutes
            priority_distribution: {"prime": 0.3, "standard": 0.5, "economy": 0.2}
        
        Returns:
            ToolResult avec le nombre de jobs ajoutés
        """
        if not self.pipeline:
            return ToolResult(success=False, message="Aucun pipeline créé. Utilisez create_pipeline d'abord.")
        
        try:
            if priority_distribution:
                for _ in range(count):
                    ptype = np.random.choice(
                        list(priority_distribution.keys()),
                        p=list(priority_distribution.values())
                    )
                    prio = {"prime": 9, "standard": 5, "economy": 2}.get(ptype, 5)
                    dl = {"prime": 60, "standard": 120, "economy": 240}.get(ptype, deadline)
                    self.pipeline.create_job(priority=prio, deadline=dl)
            else:
                for _ in range(count):
                    self.pipeline.create_job(priority=priority, deadline=deadline)
            
            self._log("add_jobs", {"count": count, "priority": priority})
            
            return ToolResult(
                success=True,
                message=f"{count} jobs ajoutés (priorité={priority}, deadline={deadline}min)",
                data={"total_jobs": self.pipeline.stats["jobs_created"]}
            )
        except Exception as e:
            return ToolResult(success=False, message=f"Erreur: {str(e)}")
    
    def run_simulation(self, steps: int = 100) -> ToolResult:
        """Exécute la simulation pour N steps.
        
        Args:
            steps: Nombre de steps à simuler
        
        Returns:
            ToolResult avec les statistiques après simulation
        """
        if not self.pipeline:
            return ToolResult(success=False, message="Aucun pipeline créé.")
        
        try:
            for _ in range(steps):
                self.pipeline.step()
            
            stats = self.pipeline.get_stats()
            self._log("run_simulation", {"steps": steps, "stats": stats})
            
            return ToolResult(
                success=True,
                message=f"Simulation de {steps} steps terminée",
                data=stats
            )
        except Exception as e:
            return ToolResult(success=False, message=f"Erreur: {str(e)}")
    
    def get_status(self) -> ToolResult:
        """Obtient le status actuel du pipeline.
        
        Returns:
            ToolResult avec les statistiques complètes
        """
        if not self.pipeline:
            return ToolResult(success=False, message="Aucun pipeline créé.")
        
        stats = self.pipeline.get_stats()
        
        # Analyse des bottlenecks
        bottlenecks = self._analyze_bottlenecks()
        
        return ToolResult(
            success=True,
            message="Status récupéré",
            data={
                **stats,
                "bottlenecks": bottlenecks
            }
        )
    
    def analyze_bottlenecks(self) -> ToolResult:
        """Analyse les bottlenecks du pipeline.
        
        Returns:
            ToolResult avec l'analyse des goulots d'étranglement
        """
        if not self.pipeline:
            return ToolResult(success=False, message="Aucun pipeline créé.")
        
        bottlenecks = self._analyze_bottlenecks()
        recommendations = self._generate_recommendations(bottlenecks)
        
        return ToolResult(
            success=True,
            message="Analyse des bottlenecks terminée",
            data={
                "bottlenecks": bottlenecks,
                "recommendations": recommendations
            }
        )
    
    def simulate_scenario(
        self,
        scenario_type: str,
        parameters: Dict
    ) -> ToolResult:
        """Simule un scénario (what-if analysis).
        
        Args:
            scenario_type: Type de scénario
                - 'add_resources': Ajouter des ressources
                - 'failure': Simuler une panne
                - 'demand_spike': Pic de demande
            parameters: Paramètres du scénario
        
        Returns:
            ToolResult avec l'impact simulé
        """
        if not self.pipeline:
            return ToolResult(success=False, message="Aucun pipeline créé.")
        
        try:
            # Sauvegarder l'état actuel
            current_stats = self.pipeline.get_stats()
            
            if scenario_type == "add_resources":
                # Simuler l'ajout de ressources
                for res_name, amount in parameters.get("resources", {}).items():
                    if res_name in self.pipeline.resources:
                        self.pipeline.resources[res_name].capacity += amount
                
                # Simuler quelques steps
                for _ in range(50):
                    self.pipeline.step()
                
                new_stats = self.pipeline.get_stats()
                
                return ToolResult(
                    success=True,
                    message="Scénario 'add_resources' simulé",
                    data={
                        "before": current_stats,
                        "after": new_stats,
                        "improvement": {
                            "completion_rate": new_stats["completion_rate"] - current_stats["completion_rate"],
                            "avg_time": current_stats["avg_time_in_system"] - new_stats["avg_time_in_system"]
                        }
                    }
                )
            
            elif scenario_type == "demand_spike":
                # Ajouter un pic de demande
                spike_count = parameters.get("count", 50)
                priority = parameters.get("priority", 5)
                
                for _ in range(spike_count):
                    self.pipeline.create_job(priority=priority, deadline=60)
                
                for _ in range(100):
                    self.pipeline.step()
                
                new_stats = self.pipeline.get_stats()
                
                return ToolResult(
                    success=True,
                    message=f"Scénario 'demand_spike' simulé (+{spike_count} jobs)",
                    data={
                        "before": current_stats,
                        "after": new_stats,
                        "impact": {
                            "jobs_failed_increase": new_stats["jobs_failed"] - current_stats["jobs_failed"],
                            "avg_time_increase": new_stats["avg_time_in_system"] - current_stats["avg_time_in_system"]
                        }
                    }
                )
            
            else:
                return ToolResult(success=False, message=f"Scénario inconnu: {scenario_type}")
                
        except Exception as e:
            return ToolResult(success=False, message=f"Erreur: {str(e)}")
    
    def set_priority(self, job_ids: List[int], new_priority: int) -> ToolResult:
        """Change la priorité de jobs.
        
        Args:
            job_ids: Liste des IDs de jobs
            new_priority: Nouvelle priorité (1-10)
        
        Returns:
            ToolResult avec le nombre de jobs modifiés
        """
        if not self.pipeline:
            return ToolResult(success=False, message="Aucun pipeline créé.")
        
        modified = 0
        for job_id in job_ids:
            if job_id in self.pipeline.jobs:
                self.pipeline.jobs[job_id].priority = new_priority
                modified += 1
        
        return ToolResult(
            success=True,
            message=f"{modified} jobs modifiés avec priorité {new_priority}",
            data={"modified_count": modified}
        )
    
    def export_gantt(self, max_jobs: int = 30) -> ToolResult:
        """Exporte les données Gantt.
        
        Args:
            max_jobs: Nombre max de jobs à exporter
        
        Returns:
            ToolResult avec les données Gantt
        """
        if not self.pipeline:
            return ToolResult(success=False, message="Aucun pipeline créé.")
        
        gantt = self.pipeline.export_gantt(max_jobs=max_jobs)
        
        return ToolResult(
            success=True,
            message=f"Gantt exporté ({len(gantt)} entrées)",
            data={"gantt": gantt}
        )
    
    # === INTERNAL METHODS ===
    
    def _analyze_bottlenecks(self) -> List[Dict]:
        """Analyse les bottlenecks du pipeline."""
        bottlenecks = []
        
        if not self.pipeline:
            return bottlenecks
        
        stats = self.pipeline.get_stats()
        
        # Vérifier l'utilisation des ressources
        for res_name, util in stats.get("resource_utilization", {}).items():
            if util > 0.8:
                bottlenecks.append({
                    "type": "resource_saturation",
                    "resource": res_name,
                    "utilization": util,
                    "severity": "high" if util > 0.95 else "medium"
                })
        
        # Vérifier le taux d'échec
        if stats["jobs_created"] > 0:
            failure_rate = stats["jobs_failed"] / stats["jobs_created"]
            if failure_rate > 0.1:
                bottlenecks.append({
                    "type": "high_failure_rate",
                    "rate": failure_rate,
                    "severity": "high" if failure_rate > 0.3 else "medium"
                })
        
        # Vérifier le temps moyen
        if stats["avg_time_in_system"] > 50:
            bottlenecks.append({
                "type": "slow_throughput",
                "avg_time": stats["avg_time_in_system"],
                "severity": "medium"
            })
        
        return bottlenecks
    
    def _generate_recommendations(self, bottlenecks: List[Dict]) -> List[str]:
        """Génère des recommandations basées sur les bottlenecks."""
        recommendations = []
        
        for bn in bottlenecks:
            if bn["type"] == "resource_saturation":
                res = bn["resource"]
                recommendations.append(
                    f"Augmenter la capacité de '{res}' (+20% recommandé)"
                )
            elif bn["type"] == "high_failure_rate":
                recommendations.append(
                    f"Étendre les deadlines ou augmenter les ressources"
                )
            elif bn["type"] == "slow_throughput":
                recommendations.append(
                    f"Optimiser les durées de stages ou ajouter du parallélisme"
                )
        
        if not recommendations:
            recommendations.append("Le pipeline fonctionne de manière optimale")
        
        return recommendations
    
    def _log(self, action: str, data: Dict):
        """Log une action."""
        self.history.append({
            "action": action,
            "data": data,
            "time": self.pipeline.current_time if self.pipeline else 0
        })
    
    # === TOOL SCHEMA FOR LLM ===
    
    def get_tools_schema(self) -> List[Dict]:
        """Retourne le schéma des tools pour le LLM."""
        return [
            {
                "name": "create_pipeline",
                "description": "Crée un nouveau pipeline de production. Types: 'fulfillment' (Amazon-like) ou 'manufacturing'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pipeline_type": {"type": "string", "enum": ["fulfillment", "manufacturing"]},
                        "robots": {"type": "integer", "description": "Nombre de robots/machines"},
                        "stations": {"type": "integer", "description": "Nombre de stations"},
                        "docks": {"type": "integer", "description": "Nombre de docks"}
                    },
                    "required": ["pipeline_type"]
                }
            },
            {
                "name": "add_jobs",
                "description": "Ajoute des jobs (commandes) au pipeline.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "count": {"type": "integer", "description": "Nombre de jobs"},
                        "priority": {"type": "integer", "description": "Priorité 1-10 (10=urgent)"},
                        "deadline": {"type": "integer", "description": "Deadline en minutes"}
                    },
                    "required": ["count"]
                }
            },
            {
                "name": "run_simulation",
                "description": "Exécute la simulation pour N steps (1 step = 1 minute).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "steps": {"type": "integer", "description": "Nombre de steps"}
                    },
                    "required": ["steps"]
                }
            },
            {
                "name": "get_status",
                "description": "Obtient le status actuel: jobs complétés, en cours, échoués, utilisation des ressources.",
                "parameters": {"type": "object", "properties": {}}
            },
            {
                "name": "analyze_bottlenecks",
                "description": "Analyse les goulots d'étranglement et génère des recommandations.",
                "parameters": {"type": "object", "properties": {}}
            },
            {
                "name": "simulate_scenario",
                "description": "Simule un scénario what-if (ajout ressources, panne, pic demande).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "scenario_type": {"type": "string", "enum": ["add_resources", "failure", "demand_spike"]},
                        "parameters": {"type": "object", "description": "Paramètres du scénario"}
                    },
                    "required": ["scenario_type", "parameters"]
                }
            },
            {
                "name": "export_gantt",
                "description": "Exporte les données du diagramme Gantt.",
                "parameters": {"type": "object", "properties": {}}
            }
        ]


# === SYSTEM PROMPT ===

SYSTEM_PROMPT = """Tu es un assistant expert en gestion de production et supply chain.
Tu contrôles un système de pipeline de production appelé LifeCore.

Tu peux:
1. Créer des pipelines (fulfillment, manufacturing)
2. Ajouter des jobs avec différentes priorités
3. Simuler et analyser les performances
4. Identifier les bottlenecks
5. Recommander des optimisations

Quand l'utilisateur demande quelque chose, utilise les tools appropriés.
Explique toujours les résultats de manière claire et actionnable.

Exemples de requêtes:
- "Crée un entrepôt avec 50 robots"
- "Ajoute 100 commandes Prime"
- "Pourquoi le taux de livraison est bas?"
- "Que se passe-t-il si on ajoute 20 robots?"
"""


# === LLM DISPATCHER ===

class LLMDispatcher:
    """Dispatche les appels de tools depuis le LLM."""
    
    def __init__(self):
        self.controller = PipelineController()
    
    def execute_tool(self, tool_name: str, parameters: Dict) -> ToolResult:
        """Exécute un tool par son nom."""
        tool_map = {
            "create_pipeline": self.controller.create_pipeline,
            "add_jobs": self.controller.add_jobs,
            "run_simulation": self.controller.run_simulation,
            "get_status": self.controller.get_status,
            "analyze_bottlenecks": self.controller.analyze_bottlenecks,
            "simulate_scenario": self.controller.simulate_scenario,
            "set_priority": self.controller.set_priority,
            "export_gantt": self.controller.export_gantt,
        }
        
        if tool_name not in tool_map:
            return ToolResult(success=False, message=f"Tool inconnu: {tool_name}")
        
        try:
            return tool_map[tool_name](**parameters)
        except Exception as e:
            return ToolResult(success=False, message=f"Erreur d'exécution: {str(e)}")
    
    def get_tools_schema(self) -> List[Dict]:
        """Retourne le schéma des tools."""
        return self.controller.get_tools_schema()
    
    def get_system_prompt(self) -> str:
        """Retourne le system prompt."""
        return SYSTEM_PROMPT


# === DEMO ===

if __name__ == "__main__":
    print("LLM Interface Demo")
    print("=" * 50)
    
    dispatcher = LLMDispatcher()
    
    # Simuler des appels comme le ferait un LLM
    print("\n1. Création du pipeline...")
    result = dispatcher.execute_tool("create_pipeline", {
        "pipeline_type": "fulfillment",
        "robots": 30,
        "stations": 15,
        "docks": 5
    })
    print(f"   {result.message}")
    
    print("\n2. Ajout de jobs...")
    result = dispatcher.execute_tool("add_jobs", {
        "count": 50,
        "priority": 5,
        "deadline": 80
    })
    print(f"   {result.message}")
    
    print("\n3. Simulation...")
    result = dispatcher.execute_tool("run_simulation", {"steps": 100})
    print(f"   {result.message}")
    if result.data:
        print(f"   Completed: {result.data['jobs_completed']}/{result.data['jobs_created']}")
    
    print("\n4. Analyse bottlenecks...")
    result = dispatcher.execute_tool("analyze_bottlenecks", {})
    print(f"   {result.message}")
    if result.data:
        for rec in result.data.get("recommendations", []):
            print(f"   → {rec}")
    
    print("\n5. Scénario: pic de demande...")
    result = dispatcher.execute_tool("simulate_scenario", {
        "scenario_type": "demand_spike",
        "parameters": {"count": 30, "priority": 9}
    })
    print(f"   {result.message}")
    
    print("\n✅ Demo terminée!")
    print(f"\nTools disponibles: {[t['name'] for t in dispatcher.get_tools_schema()]}")
