# LifeCore V3

**Framework d'orchestration hiérarchique pour systèmes adaptatifs**

## Ce que c'est

LifeCore modélise des systèmes hiérarchiques où:
- Chaque **node** a des besoins et des objectifs
- Les **enfants** reportent leurs contraintes au parent
- Les **ressources** sont partagées et allouées par priorité
- La **stratégie émerge** de l'expérience (bonnes expériences → comportement optimisé)

## Concepts

```
LifeCore Node
├── Goals     → Ce qu'on veut atteindre
├── Needs     → Besoins homéostatiques → génèrent des intentions
├── Memory    → Stocke les expériences, réutilise les bonnes
├── Resources → Partagées, limitées, allouées par priorité
├── Laws      → Contraintes externes (limites, zones)
├── Children  → Sous-systèmes avec leurs propres besoins
└── Feedback  → Contraintes qui remontent des enfants
```

## Installation

```bash
pip install numpy
```

## Usage

```python
from lifecore import LifeCore, Need, Goal
import numpy as np

# Hiérarchie: Controller → Motors
controller = LifeCore(dims=4)
motor_l = controller.spawn_child(domain_dims=[0, 1])
motor_r = controller.spawn_child(domain_dims=[2, 3])

# Les enfants ont des contraintes
# Le parent reçoit le feedback
intention = controller.get_recursive_intention(state)
```

## Configuration YAML/JSON

```yaml
name: "Drone Fleet"
dims: 7
resources:
  - name: battery
    capacity: 1000
hierarchy:
  name: controller
  children:
    - name: drone
      count: 5
```

```python
from lifecore.config import load_system
system = load_system("config.yaml")
```

## Modules

| Module | Lignes | Description |
|--------|--------|-------------|
| core.py | ~430 | Agent fractal avec feedback récursif |
| memory.py | 238 | Mémoire tensorielle, réutilisation directe |
| resource.py | 218 | Ressources partagées, allocation par priorité |
| config.py | 338 | Loader YAML/JSON |
| law.py | 185 | Contraintes (vitesse, zones, murs) |
| capability.py | 180 | Limites internes (saturation douce) |
| goal.py | 157 | Objectifs |
| need.py | 112 | Besoins homéostatiques |
| activation.py | 161 | Fonctions smooth (sigmoid) |
| coherence.py | 189 | Couplage entre frères |

## Principe clé

La **stratégie n'est pas programmée** - elle émerge de l'expérience:
- Bonnes expériences → mémorisées
- Situations similaires → comportement réutilisé
- Optimisation globale via allocation de ressources

## License

MIT
