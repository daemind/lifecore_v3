#!/usr/bin/env python3
"""
LifeCore V3 - MuJoCo TaskLearner Integration
=============================================

Connects LifeCore's goal-based TaskLearner to Franka Panda in MuJoCo.
Demonstrates learned pick-and-place with REAL physics.

Usage:
    source venv311/bin/activate
    cd models/franka_emika_panda
    mjpython ../../lifecore/sim_mujoco_tasklearner.py
"""

import numpy as np
from typing import Dict, List, Optional
import time
import os
import sys

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

from lifecore.core import LifeCore, Need


class MuJoCoScene:
    """Scene MuJoCo avec Franka Panda + objets.
    
    Adapte l'interface Scene de robotics.py pour MuJoCo.
    """
    
    # Scene XML with arm + objects
    SCENE_XML = """
    <mujoco model="panda_pickplace">
      <include file="panda.xml"/>
      
      <worldbody>
        <!-- Table -->
        <geom name="table" type="box" pos="0.5 0 0.2" size="0.3 0.4 0.02" 
              rgba="0.6 0.4 0.2 1" contype="1" conaffinity="1"/>
        
        <!-- Ball to pick -->
        <body name="ball" pos="0.5 0.15 0.25">
          <freejoint name="ball_free"/>
          <geom name="ball_geom" type="sphere" size="0.025" rgba="1 0.2 0.2 1" 
                mass="0.05" contype="1" conaffinity="1"/>
        </body>
        
        <!-- Target glass -->
        <body name="glass" pos="0.5 -0.15 0.22">
          <geom name="glass_bottom" type="cylinder" size="0.04 0.005" rgba="0.3 0.3 0.8 0.7"/>
        </body>
      </worldbody>
    </mujoco>
    """
    
    def __init__(self, visualize: bool = True):
        if not MUJOCO_AVAILABLE:
            raise RuntimeError("MuJoCo not installed")
        
        # Load model from panda directory
        base_dir = os.path.dirname(os.path.dirname(__file__))
        panda_dir = os.path.join(base_dir, "models", "franka_emika_panda")
        
        # Use panda.xml directly (it has scene already)
        model_path = os.path.join(panda_dir, "pick_place_scene.xml")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        self.visualize = visualize
        self.viewer = None
        
        # Joint indices
        self.arm_joints = list(range(7))
        self.gripper_joints = [7, 8]  # finger joints
        
        # Control
        self.kp = np.array([600, 600, 600, 600, 250, 150, 50])
        self.kd = np.array([50, 50, 50, 50, 30, 25, 15])
        
        # Initial pose: arm extended forward, gripper above table
        # Table is at z=0.22, so gripper needs to be at z > 0.3
        self.home_qpos = np.array([
            0.0,      # shoulder pan (centered)
            0.2,      # shoulder lift (slightly raised)
            0.0,      # upper arm roll
            -1.5,     # elbow (less bent to keep higher)
            0.0,      # forearm roll
            1.5,      # wrist pitch
            0.785     # wrist roll
        ])
        self.target_qpos = self.home_qpos.copy()
        self.gripper_open = True
        
        # Set initial position
        for i, q in enumerate(self.home_qpos):
            self.data.qpos[i] = q
        mujoco.mj_forward(self.model, self.data)
        
        # Simulated objects (since we can't easily add freejoint bodies)
        self.objects = {
            "ball": {"pos": np.array([0.5, 0.15, 0.27]), "held": False, "size": 0.025},
            "glass": {"pos": np.array([0.5, -0.15, 0.22]), "is_container": True}
        }
        self.held_object = None
        
        print(f"MuJoCoScene initialized with {self.model.njnt} joints")
        print(f"Initial EE position: {self.ee_pos.round(3)}")
    
    def start_viewer(self):
        if self.visualize and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
    
    def close_viewer(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
    @property
    def joint_positions(self) -> np.ndarray:
        return self.data.qpos[:7].copy()
    
    @property
    def joint_velocities(self) -> np.ndarray:
        return self.data.qvel[:7].copy()
    
    @property
    def ee_pos(self) -> np.ndarray:
        """End effector position."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        if body_id < 0:
            body_id = 7  # Fallback to last link
        return self.data.xpos[body_id].copy()
    
    @property
    def gripper_is_holding(self) -> bool:
        return self.held_object is not None
    
    def set_targets(self, qpos: np.ndarray):
        self.target_qpos = np.clip(qpos, -2.9, 2.9)
    
    def open_gripper(self):
        self.gripper_open = True
        if self.held_object:
            # Drop object at current EE position
            self.objects[self.held_object]["pos"] = self.ee_pos.copy()
            self.objects[self.held_object]["pos"][2] -= 0.05
            self.objects[self.held_object]["held"] = False
            self.held_object = None
    
    def close_gripper(self):
        self.gripper_open = False
        # Check if near any object
        for name, obj in self.objects.items():
            if obj.get("is_container"):
                continue
            if not obj["held"]:
                dist = np.linalg.norm(self.ee_pos - obj["pos"])
                if dist < 0.08:
                    self.held_object = name
                    obj["held"] = True
                    break
    
    def step(self, n_substeps: int = 5):
        for _ in range(n_substeps):
            # PD control
            qpos = self.joint_positions
            qvel = self.joint_velocities
            error = self.target_qpos - qpos
            ctrl = self.kp * error - self.kd * qvel
            
            for i in range(min(7, self.model.nu)):
                self.data.ctrl[i] = ctrl[i]
            
            # Gripper: actuator index 7, range [0, 255]
            # High value = open, low value = closed
            grip_ctrl = 255.0 if self.gripper_open else 0.0
            if self.model.nu > 7:
                self.data.ctrl[7] = grip_ctrl
            
            mujoco.mj_step(self.model, self.data)
        
        # Update held object position
        if self.held_object:
            self.objects[self.held_object]["pos"] = self.ee_pos.copy()
        
        if self.viewer and self.viewer.is_running():
            self.viewer.sync()
    
    def move_to_position(self, target: np.ndarray, max_steps: int = 200) -> bool:
        """Move end effector to target position using IK."""
        for _ in range(max_steps):
            mujoco.mj_forward(self.model, self.data)
            
            error = target - self.ee_pos
            if np.linalg.norm(error) < 0.03:
                return True
            
            # Simple Jacobian IK
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
            if body_id < 0:
                body_id = 7
            
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
            
            J = jacp[:, :7]
            damping = 0.01
            JJT = J @ J.T + damping * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, error * 0.5)
            
            new_qpos = self.joint_positions + np.clip(dq, -0.1, 0.1)
            self.set_targets(new_qpos)
            self.step(10)
        
        return False
    
    def is_ball_in_glass(self) -> bool:
        """Check if ball is in glass."""
        ball = self.objects.get("ball")
        glass = self.objects.get("glass")
        if not ball or not glass:
            return False
        
        dist_xy = np.linalg.norm(ball["pos"][:2] - glass["pos"][:2])
        return dist_xy < 0.04 and ball["pos"][2] < glass["pos"][2] + 0.1


class MuJoCoTaskLearner:
    """TaskLearner connected to MuJoCo physics."""
    
    def __init__(self, scene: MuJoCoScene):
        self.scene = scene
        
        # LifeCore for learning
        self.core = LifeCore(dims=3)
        
        # Goal need
        self.goal_need = Need(
            sub_matrix=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            extractor=lambda s: 1.0 - s[2],
            urgency_fn=lambda d: float(np.clip(d, 0, 1)),
            priority=3.0,
            name="goal_achievement"
        )
        self.core.needs.append(self.goal_need)
        
        # Action history
        self.actions_taken = []
    
    def get_state(self) -> np.ndarray:
        """Get current state for LifeCore."""
        holding = 1.0 if self.scene.gripper_is_holding else 0.0
        
        # Distance to nearest object
        min_dist = 1.0
        for name, obj in self.scene.objects.items():
            if not obj.get("is_container") and not obj.get("held"):
                dist = np.linalg.norm(self.scene.ee_pos - obj["pos"])
                min_dist = min(min_dist, dist)
        
        # Goal progress
        goal_progress = 1.0 if self.scene.is_ball_in_glass() else 0.0
        
        return np.array([holding, 1.0 - min_dist, goal_progress], dtype=np.float32)
    
    def get_possible_actions(self) -> List[Dict]:
        """Get available actions."""
        actions = []
        
        if self.scene.gripper_open:
            actions.append({"type": "close_gripper"})
        else:
            actions.append({"type": "open_gripper"})
        
        for name, obj in self.scene.objects.items():
            if not obj.get("held"):
                actions.append({"type": "goto", "target": name})
        
        return actions
    
    def score_action(self, action: Dict, state: np.ndarray) -> float:
        """Score an action based on current state."""
        score = 0.0
        holding = state[0] > 0.5
        
        # Phase 1: Not holding - go to ball and grab
        if not holding:
            if action["type"] == "goto" and action.get("target") == "ball":
                score += 5.0
            elif action["type"] == "close_gripper":
                # Good if near ball
                ball_pos = self.scene.objects["ball"]["pos"]
                dist = np.linalg.norm(self.scene.ee_pos - ball_pos)
                if dist < 0.1:
                    score += 8.0
        
        # Phase 2: Holding - go to glass and drop
        else:
            if action["type"] == "goto" and action.get("target") == "glass":
                score += 5.0
            elif action["type"] == "open_gripper":
                glass_pos = self.scene.objects["glass"]["pos"]
                dist = np.linalg.norm(self.scene.ee_pos - glass_pos)
                if dist < 0.1:
                    score += 8.0
        
        # Add LifeCore influence
        intention = self.core.get_intention(state)
        score += float(np.dot(intention, state)) * 0.3
        
        return score
    
    def execute_action(self, action: Dict) -> bool:
        """Execute an action."""
        if action["type"] == "close_gripper":
            self.scene.close_gripper()
            return True
        
        elif action["type"] == "open_gripper":
            self.scene.open_gripper()
            return True
        
        elif action["type"] == "goto":
            target_name = action["target"]
            obj = self.scene.objects.get(target_name)
            if obj:
                target_pos = obj["pos"].copy()
                target_pos[2] += 0.08  # Go above
                return self.scene.move_to_position(target_pos)
        
        return False
    
    def run_episode(self, max_actions: int = 20) -> Dict:
        """Run a learning episode."""
        self.scene.start_viewer()
        
        print("MuJoCo TaskLearner Episode")
        print("=" * 40)
        print(f"Ball: {self.scene.objects['ball']['pos'].round(3)}")
        print(f"Glass: {self.scene.objects['glass']['pos'].round(3)}")
        print(f"EE: {self.scene.ee_pos.round(3)}")
        print()
        
        actions_taken = 0
        success = False
        action_log = []
        
        for step in range(max_actions):
            if self.scene.viewer and not self.scene.viewer.is_running():
                break
            
            # Check goal
            if self.scene.is_ball_in_glass():
                success = True
                print("🎉 GOAL ACHIEVED: Ball is in glass!")
                break
            
            # Get state and actions
            state = self.get_state()
            actions = self.get_possible_actions()
            
            # Score and select best action
            scored = [(a, self.score_action(a, state)) for a in actions]
            scored.sort(key=lambda x: -x[1])
            
            best_action = scored[0][0]
            
            print(f"Step {step + 1}: {best_action} (score: {scored[0][1]:.2f})")
            
            # Execute
            self.execute_action(best_action)
            action_log.append(best_action)
            actions_taken += 1
            
            # Learn from experience
            new_state = self.get_state()
            quality = new_state[2] - state[2]  # Goal progress delta
            intention = self.core.get_intention(state)
            self.core.add_experience(state, intention, new_state - state, quality)
            
            # Step physics
            for _ in range(30):
                self.scene.step(5)
                time.sleep(0.02)
        
        print()
        print(f"Episode complete: {actions_taken} actions, Success: {success}")
        print(f"Final ball: {self.scene.objects['ball']['pos'].round(3)}")
        
        self.scene.close_viewer()
        
        return {
            "success": success,
            "actions_taken": actions_taken,
            "action_log": action_log
        }


def main():
    if not MUJOCO_AVAILABLE:
        print("MuJoCo not available")
        return
    
    scene = MuJoCoScene(visualize=True)
    learner = MuJoCoTaskLearner(scene)
    
    result = learner.run_episode(max_actions=15)
    
    print()
    print("=" * 40)
    print(f"Result: {'SUCCESS' if result['success'] else 'FAILED'}")
    print(f"Actions: {result['actions_taken']}")


if __name__ == "__main__":
    main()
