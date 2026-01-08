# LifeCore V3

**Framework d'orchestration hiérarchique pour pipelines de production**

## Ce que c'est

LifeCore modélise des **pipelines de production dynamiques** (Gantt adaptatif) où:
- Les **stages** ont des durées et des ressources paramétrables
- Les **jobs** traversent le pipeline selon les dépendances
- Les **ressources** sont partagées et allouées par priorité
- La **stratégie émerge** de l'expérience (optimisation continue)

## Architecture

```
Pipeline (Gantt dynamique)
├── Stages    → Étapes avec durée, ressources, variabilité
├── Jobs      → Instances traversant le pipeline (priorité, deadline)
├── Resources → Pools partagés (robots, stations, docks)
├── Events    → Alertes, pannes, demand spikes
└── Gantt     → Export données pour visualisation
```

## Module Pipeline

Le cœur de LifeCore pour le scheduling de production:

```python
from lifecore.pipeline import Pipeline, Stage

# Définir un pipeline de fulfillment
pipe = Pipeline("fulfillment")

# Stages paramétrables
pipe.add_stage(Stage("picking", duration=5, resources={"robot": 1}))
pipe.add_stage(Stage("transport", duration=3, resources={"robot": 1}))
pipe.add_stage(Stage("packing", duration=2, resources={"station": 1}))
pipe.add_stage(Stage("shipping", duration=1, resources={"dock": 1}))

# Dépendances (DAG)
pipe.add_dependency("picking", "transport")
pipe.add_dependency("transport", "packing")
pipe.add_dependency("packing", "shipping")

# Capacités ressources
pipe.set_resource("robot", capacity=50)
pipe.set_resource("station", capacity=20)
pipe.set_resource("dock", capacity=10)

# Créer des jobs avec priorité et deadline
for i in range(100):
    pipe.create_job(priority=9, deadline=60)

# Simuler
pipe.run(duration=200)

# Export Gantt
gantt_data = pipe.export_gantt()
print(pipe.get_stats())
```

**Résultat:**
```
Pipeline: fulfillment
Completed: 100/100
Avg time: 13.6 min
Utilization: robot=15%, station=12%, dock=8%
```

## Pipelines prédefinis

```python
from lifecore.pipeline import create_fulfillment_pipeline, create_manufacturing_pipeline

# Amazon-style fulfillment
pipe = create_fulfillment_pipeline(robots=200, packing_stations=50, shipping_docks=10)

# Manufacturing line
pipe = create_manufacturing_pipeline(machines=10, workers=20, testers=3)
```

## Applications Supply Chain

| Supply Chain | LifeCore Pipeline |
|--------------|-------------------|
| Étapes process | Stages avec durée |
| Capacité machines | ResourcePool |
| Priorité commandes | Job.priority |
| SLA / Délais | Job.deadline |
| Dépendances | DAG de stages |
| Visualisation | export_gantt() |

### Exemples inclus

- **Amazon Fulfillment** - 200 robots, 50 stations, 5000 commandes
- **Drone Delivery Fleet** - 270 drones, 9 zones, 1000 livraisons
- **Gradio Dashboard** - Visualisation temps réel

```bash
# Lancer la simulation Amazon
python examples/amazon_fulfillment_simulation.py

# Lancer le dashboard drones
python examples/drone_fleet_gradio.py

# Tester le module Pipeline
python lifecore/pipeline.py
```

## Système d'événements

Gestion des situations dynamiques:

```python
from lifecore.event import EventBus, battery_low_alert, equipment_failure

bus = EventBus()
bus.subscribe_type(EventType.ALERT, handler)

# Émettre des événements
bus.emit(battery_low_alert("drone_42", level=0.15))
bus.emit(equipment_failure("robot_7", "motor"))
```

## Installation

```bash
pip install numpy pyyaml
# Pour le dashboard:
pip install gradio
```

## Modules

| Module | Lignes | Description |
|--------|--------|-------------|
| **pipeline.py** | 570 | Pipeline scheduling / Gantt dynamique |
| core.py | 430 | Agent fractal avec feedback récursif |
| event.py | 363 | EventBus pour alertes et pannes |
| memory.py | 238 | Mémoire tensorielle |
| resource.py | 218 | Ressources partagées |
| config.py | 338 | Loader YAML/JSON |
| law.py | 185 | Contraintes (vitesse, zones) |

**Total: ~6000 lignes**

## Tests

```bash
# Lancer tous les tests (33 tests)
python tests/test_all.py
```

## Principe clé

**LifeCore = Gantt dynamique** qui s'adapte en temps réel:
- Les jobs sont schedulés par priorité
- Les ressources sont allouées dynamiquement
- Les deadlines sont enforced
- Les événements sont gérés asynchronement

## Roadmap

### ✅ Terminé
- [x] Module Pipeline (Gantt dynamique)
- [x] Système d'événements (alertes, pannes)
- [x] Tests unitaires (33 tests)
- [x] Simulations réalistes (Amazon, Drones)

### En cours
- [ ] Visualisation Gantt (Gradio/HTML)
- [ ] Prédiction de demande
- [ ] Multi-objectif (coût vs délai vs qualité)

### Long terme
- [ ] Digital Twin - Synchronisation temps réel
- [ ] Intégration ERP
- [ ] Explainability - Pourquoi cette décision

## License

MIT
