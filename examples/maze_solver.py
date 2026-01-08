#!/usr/bin/env python3
"""
MAZE SOLVER - LifeCore V3 avec Stratégie
========================================

Test du système de stratégie sur un maze complexe.

Maze:
    S . . # . . . . . .
    . # . # . # # # # .
    . # . . . . . . # .
    . # # # # # . . # .
    . . . . . # . . . .
    # # # # . # . # # #
    . . . . . # . . . E

S = Start (0, 0)
E = Exit (9, 6)
# = Mur
. = Passage

Run:
    cd lifecore-v3-clean
    python examples/maze_solver.py
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from lifecore import LifeCore, Goal
from lifecore.strategy import create_maze_strategy, AStarStrategy


# === MAZE DEFINITION ===

MAZE = [
    "S . . # . . . . . .",
    ". # . # . # # # # .",
    ". # . . . . . . # .",
    ". # # # # # . . # .",
    ". . . . . # . . . .",
    "# # # # . # . # # #",
    ". . . . . # . . . E",
]

def parse_maze(maze_lines):
    """Parse le maze en liste d'obstacles."""
    obstacles = []
    start = None
    end = None
    
    for y, line in enumerate(maze_lines):
        cells = line.split()
        for x, cell in enumerate(cells):
            if cell == '#':
                obstacles.append(np.array([x, y, 0]))
            elif cell == 'S':
                start = np.array([x, y, 0.0, 0.0])
            elif cell == 'E':
                end = np.array([x, y, 0.0, 0.0])
    
    return obstacles, start, end


def print_maze_with_path(maze_lines, path, current_pos=None):
    """Affiche le maze avec le chemin parcouru."""
    # Créer une grille modifiable
    grid = []
    for line in maze_lines:
        cells = line.split()
        grid.append(cells)
    
    # Marquer le chemin
    for pos in path:
        x, y = int(round(pos[0])), int(round(pos[1]))
        if 0 <= y < len(grid) and 0 <= x < len(grid[0]):
            if grid[y][x] not in ['S', 'E', '#']:
                grid[y][x] = '·'
    
    # Marquer la position actuelle
    if current_pos is not None:
        x, y = int(round(current_pos[0])), int(round(current_pos[1]))
        if 0 <= y < len(grid) and 0 <= x < len(grid[0]):
            if grid[y][x] not in ['S', 'E', '#']:
                grid[y][x] = '@'
    
    # Afficher
    print("+" + "-" * 21 + "+")
    for row in grid:
        print("|", ' '.join(row), "|")
    print("+" + "-" * 21 + "+")


def run_maze_solver():
    print("=" * 60)
    print("  MAZE SOLVER - LifeCore V3 Strategy")
    print("=" * 60)
    print()
    
    # Parser le maze
    obstacles, start, end = parse_maze(MAZE)
    print(f"Départ: ({start[0]:.0f}, {start[1]:.0f})")
    print(f"Sortie: ({end[0]:.0f}, {end[1]:.0f})")
    print(f"Obstacles: {len(obstacles)} murs")
    print()
    
    print("Maze initial:")
    print_maze_with_path(MAZE, [])
    print()
    
    # Créer l'agent avec stratégie A*
    agent = LifeCore(dims=4)
    agent.set_strategy(AStarStrategy(grid_size=1.0))
    
    # Définir l'objectif
    agent.goals.push(Goal(
        target=end,
        name='exit_maze',
        priority=10.0,
        threshold=0.5
    ))
    
    # Simulation
    state = start.copy()
    path = [state[:2].copy()]
    max_steps = 100
    
    print("Résolution...")
    print("-" * 40)
    
    for step in range(max_steps):
        intention = agent.get_strategic_intention(state, obstacles)
        
        # Appliquer le mouvement
        movement = intention[:2] * 0.5
        new_pos = state[:2] + movement
        
        # Vérifier collision
        is_blocked = False
        for obs in obstacles:
            if np.linalg.norm(new_pos - obs[:2]) < 0.5:
                is_blocked = True
                break
        
        if not is_blocked:
            state[:2] = new_pos
            path.append(state[:2].copy())
        
        # Vérifier si arrivé
        dist = np.linalg.norm(state[:2] - end[:2])
        
        if step % 10 == 0:
            print(f"  Step {step}: pos=({state[0]:.1f}, {state[1]:.1f}) dist={dist:.1f}")
        
        if dist < 1.0:
            print()
            print("=" * 60)
            print(f"  ✅ MAZE RÉSOLU en {step} steps!")
            print("=" * 60)
            print()
            print("Chemin trouvé:")
            print_maze_with_path(MAZE, path, state[:2])
            return True
    
    print()
    print("⚠️ Pas trouvé de solution en", max_steps, "steps")
    print_maze_with_path(MAZE, path, state[:2])
    return False


if __name__ == "__main__":
    run_maze_solver()
