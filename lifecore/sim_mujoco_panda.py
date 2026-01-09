#!/usr/bin/env python3
"""
LifeCore V3 - MuJoCo Simulator with Franka Panda
=================================================

Uses official Franka Panda model from MuJoCo Menagerie.
Demonstrates pick-and-place with real physics.

Setup:
    source venv311/bin/activate
    cd lifecore-v3-clean
    mjpython lifecore/sim_mujoco_panda.py
"""

import numpy as np
from typing import Dict, Optional
import time
import os

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("MuJoCo not installed. Run: pip install mujoco")


class FrankaPandaArm:
    """Franka Panda arm with MuJoCo physics.
    
    Uses official model from MuJoCo Menagerie.
    """
    
    def __init__(self, model_path: str = None, visualize: bool = True):
        if not MUJOCO_AVAILABLE:
            raise RuntimeError("MuJoCo not installed")
        
        # Find model path
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            model_path = os.path.join(base_dir, "models", "franka_emika_panda", "panda.xml")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Viewer
        self.visualize = visualize
        self.viewer = None
        
        # Joint info (7 arm + 2 gripper)
        self.arm_joint_names = [
            "joint1", "joint2", "joint3", "joint4", 
            "joint5", "joint6", "joint7"
        ]
        self.gripper_joint_names = ["finger_joint1", "finger_joint2"]
        
        # Get joint IDs
        self.arm_joint_ids = []
        for name in self.arm_joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                self.arm_joint_ids.append(jid)
        
        self.gripper_joint_ids = []
        for name in self.gripper_joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                self.gripper_joint_ids.append(jid)
        
        # Control gains
        self.kp = np.array([600, 600, 600, 600, 250, 150, 50])  # Position gains
        self.kd = np.array([50, 50, 50, 50, 30, 25, 15])  # Velocity gains
        
        # Targets
        self.target_qpos = np.zeros(7)
        self.gripper_target = 0.04  # Open (0.04 = max open, 0 = closed)
        
        # Initial neutral pose
        self.home_qpos = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
        self.target_qpos = self.home_qpos.copy()
        
        print(f"Franka Panda loaded: {len(self.arm_joint_ids)} arm joints, {len(self.gripper_joint_ids)} gripper joints")
    
    def start_viewer(self):
        """Start MuJoCo viewer."""
        if self.visualize and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
    
    def close_viewer(self):
        """Close viewer."""
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
    @property
    def joint_positions(self) -> np.ndarray:
        """Get arm joint positions."""
        return np.array([self.data.qpos[i] for i in self.arm_joint_ids])
    
    @property
    def joint_velocities(self) -> np.ndarray:
        """Get arm joint velocities."""
        return np.array([self.data.qvel[i] for i in self.arm_joint_ids])
    
    @property
    def end_effector_pos(self) -> np.ndarray:
        """Get end effector position."""
        # Use last link body
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        if body_id < 0:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link7")
        return self.data.xpos[body_id].copy()
    
    def set_joint_targets(self, targets: np.ndarray):
        """Set target joint positions."""
        self.target_qpos = np.clip(targets, -2.9, 2.9)
    
    def set_gripper(self, width: float):
        """Set gripper opening (0=closed, 0.04=open)."""
        self.gripper_target = np.clip(width, 0.0, 0.04)
    
    def open_gripper(self):
        """Open gripper."""
        self.set_gripper(0.04)
    
    def close_gripper(self):
        """Close gripper."""
        self.set_gripper(0.0)
    
    def go_home(self):
        """Move to home position."""
        self.set_joint_targets(self.home_qpos)
    
    def step(self, n_substeps: int = 5):
        """Step simulation with PD control."""
        for _ in range(n_substeps):
            # PD control for arm
            qpos = self.joint_positions
            qvel = self.joint_velocities
            
            error = self.target_qpos - qpos
            ctrl = self.kp * error - self.kd * qvel
            
            # Apply to actuators (first 7)
            for i in range(min(7, self.model.nu)):
                self.data.ctrl[i] = ctrl[i] if i < len(ctrl) else 0
            
            # Gripper control (last 2 actuators if present)
            if self.model.nu > 7:
                self.data.ctrl[7] = self.gripper_target * 255  # Scaled
                if self.model.nu > 8:
                    self.data.ctrl[8] = self.gripper_target * 255
            
            mujoco.mj_step(self.model, self.data)
        
        # Sync viewer
        if self.viewer and self.viewer.is_running():
            self.viewer.sync()
    
    def run_demo(self, duration: float = 10.0):
        """Run a simple movement demo."""
        self.start_viewer()
        
        print("Franka Panda Demo")
        print("=" * 40)
        print(f"End effector: {self.end_effector_pos.round(3)}")
        print()
        
        dt = 0.02
        steps = int(duration / dt)
        
        # Go to home first
        self.go_home()
        
        phase = 0
        phase_start = 0
        
        for step in range(steps):
            if self.viewer and not self.viewer.is_running():
                break
            
            t = step * dt
            
            # Phase 0: Wait and go home
            if phase == 0 and t > 1.0:
                print("Phase 1: Moving to ready position")
                # Bend down
                ready_pos = self.home_qpos.copy()
                ready_pos[1] = 0.0  # Shoulder up
                ready_pos[3] = -1.5  # Elbow
                ready_pos[5] = 1.8  # Wrist
                self.set_joint_targets(ready_pos)
                self.open_gripper()
                phase = 1
                phase_start = t
            
            elif phase == 1 and t - phase_start > 2.0:
                print("Phase 2: Reaching forward")
                reach_pos = self.target_qpos.copy()
                reach_pos[0] = 0.3  # Base rotation
                reach_pos[1] = 0.2  # Shoulder
                reach_pos[3] = -2.0  # Lower elbow
                self.set_joint_targets(reach_pos)
                phase = 2
                phase_start = t
            
            elif phase == 2 and t - phase_start > 2.0:
                print("Phase 3: Closing gripper")
                self.close_gripper()
                phase = 3
                phase_start = t
            
            elif phase == 3 and t - phase_start > 1.0:
                print("Phase 4: Lifting")
                lift_pos = self.target_qpos.copy()
                lift_pos[1] = -0.5  # Lift shoulder
                lift_pos[3] = -1.5  # Raise elbow
                self.set_joint_targets(lift_pos)
                phase = 4
                phase_start = t
            
            elif phase == 4 and t - phase_start > 2.0:
                print("Phase 5: Opening gripper")
                self.open_gripper()
                phase = 5
                phase_start = t
            
            elif phase == 5 and t - phase_start > 1.0:
                print("Done!")
                break
            
            self.step(5)
            time.sleep(dt)
        
        print(f"\nFinal EE position: {self.end_effector_pos.round(3)}")
        self.close_viewer()


def main():
    """Run Franka Panda demo."""
    if not MUJOCO_AVAILABLE:
        print("MuJoCo not available")
        return
    
    arm = FrankaPandaArm(visualize=True)
    arm.run_demo(duration=12.0)


if __name__ == "__main__":
    main()
