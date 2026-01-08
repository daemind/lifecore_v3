# LifeCore V3

**Agent adaptatif fractal avec planification**

Un framework générique pour systèmes autonomes: drones, usines, voitures, robots...

## 🎯 Concepts Clés

```
┌─────────────────────────────────────────────────────────────┐
│                      LifeCore Node                          │
├─────────────────────────────────────────────────────────────┤
│  GOALS      → Objectifs à atteindre                         │
│  NEEDS      → Besoins homéostatiques (génèrent intentions) │
│  STRATEGY   → Planification (A*, exploration, backtrack)    │
│  MEMORY     → Réutilisation directe des expériences         │
│  CHILDREN   → Sous-systèmes fractals                        │
│  RESOURCES  → Ressources partagées limitées                 │
│  LAWS       → Contraintes externes (murs, vitesse...)       │
│  CAPABILITY → Limites internes (vitesse max moteur...)      │
├─────────────────────────────────────────────────────────────┤
│                    FLUX PRINCIPAL                           │
│  Goal → Strategy → Intention → Capacity Feedback → Effect   │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Installation

```bash
pip install numpy
# Optionnel pour configs YAML:
pip install pyyaml
```

## 🚀 Quick Start

```python
from lifecore import LifeCore, Goal, Need
import numpy as np

# Créer un agent
agent = LifeCore(dims=4)

# Ajouter un objectif
agent.goals.push(Goal(
    target=np.array([10, 10, 0, 0]),
    name="reach_target"
))

# Obtenir l'intention
state = np.array([0, 0, 0, 0])
intention = agent.get_intention(state)
```

## 📁 Structure

```
lifecore-v3-clean/
├── lifecore/
│   ├── core.py        (493 lignes) Agent principal + feedback récursif
│   ├── strategy.py    (332 lignes) 6 stratégies de planification
│   ├── config.py      (338 lignes) Loader YAML/JSON
│   ├── memory.py      (238 lignes) Mémoire tensorielle
│   ├── resource.py    (218 lignes) Ressources partagées
│   ├── coherence.py   (189 lignes) Couplage entre frères
│   ├── law.py         (185 lignes) Contraintes externes
│   ├── capability.py  (180 lignes) Capacités internes
│   ├── activation.py  (161 lignes) Fonctions smooth (sigmoid, relu)
│   ├── goal.py        (157 lignes) Objectifs
│   └── need.py        (112 lignes) Besoins homéostatiques
│
├── configs/
│   ├── drone_delivery.yaml    Service de livraison drone
│   ├── autonomous_car.yaml    Voiture autonome
│   └── factory.yaml           Ligne de production
│
├── examples/
│   ├── maze_solver.py         Résolution de maze
│   ├── drone_delivery_service.py
│   ├── rocket_resources.py    Démonstration émergence
│   └── ...
│
└── tests/
    └── test_lifecore.py
```

## 🔧 Configuration Paramétrique

Définir un système entier en JSON/YAML:

```json
{
  "name": "Drone Fleet",
  "dims": 7,
  "resources": [{"name": "battery", "capacity": 1000}],
  "laws": [{"type": "speed_limit", "max": 15}],
  "hierarchy": {
    "name": "controller",
    "children": [
      {"name": "drone", "count": 5, "domain": [0,1,2]}
    ]
  }
}
```

```python
from lifecore.config import load_system
system = load_system("configs/drone_delivery.json")
```

## 🧠 Stratégies Disponibles

| Stratégie | Description |
|-----------|-------------|
| `DirectStrategy` | Ligne droite vers la cible |
| `ExplorationStrategy` | Essayer différentes directions |
| `BacktrackStrategy` | Revenir en arrière si bloqué |
| `DecomposeStrategy` | Diviser en waypoints |
| `AStarStrategy` | Planification de chemin |
| `CompositeStrategy` | Combiner plusieurs stratégies |

## 🔄 Feedback Récursif

Les contraintes remontent du bas vers le haut:

```python
# Parent demande vitesse 10
intention = parent.get_intention(state)  # [10, 0, 0]

# Enfant (moteur) reporte ce qu'il peut faire
capacity = child.get_capacity(intention)  # [8, 0, 0] (limité)

# Parent ajuste son intention
adjusted = parent.get_recursive_intention(state)  # [8, 0, 0]
```

## ✅ Ce qui fonctionne

- [x] Architecture fractale (parent → enfants)
- [x] Besoins → intentions
- [x] Mémoire tensorielle avec réutilisation
- [x] Goals et GoalStack
- [x] Ressources partagées avec allocation par priorité
- [x] Lois (vitesse, murs, zones interdites, feux)
- [x] Capacités internes (saturation douce)
- [x] Configuration YAML/JSON
- [x] Stratégies basiques (exploration, backtrack)
- [x] Feedback récursif des contraintes
- [x] Activation functions (sigmoid, smooth_threshold)

## 🚧 TODO - Prochaines Étapes

### Court terme
- [ ] **Vrai A*** - Pré-calculer le chemin complet avant de bouger
- [ ] **Cohérence automatique** - CoherenceNeed intégré sans config manuelle
- [ ] **Tests unitaires complets** - Couvrir tous les modules

### Moyen terme
- [ ] **Mazes plus grands** - 50x50+ pour voir la stratégie émerger
- [ ] **Apprentissage de stratégie** - Mémoriser des patterns de maze
- [ ] **Simulation temps réel** - Visualisation graphique
- [ ] **Multi-agent coordination** - Plusieurs LifeCore qui collaborent

### Long terme
- [ ] **Meta-learning** - Apprendre à résoudre des mazes, pas juste ce maze
- [ ] **Hiérarchie dynamique** - Créer/supprimer des enfants selon les besoins
- [ ] **LLM integration** - Intentions en langage naturel
- [ ] **Déploiement hardware** - Drones réels, robots

## 📄 License

MIT

## 👥 Auteur

BioMatrix-MVA / LifeCore Team
