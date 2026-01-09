#!/usr/bin/env python3
"""
LifeCore V3 - MuJoCo Simulator Connector
=========================================

Connecte le module robotics de LifeCore au simulateur physique MuJoCo.
Permet de valider le comportement du bras robotique avec une vraie physique.

Usage:
    # Activation du venv avec MuJoCo:
    source venv311/bin/activate
    
    # Run:
    python examples/sim_mujoco_demo.py

Requirements:
    - Python 3.11 (MuJoCo wheels)
    - mujoco >= 3.0
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import time

try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("MuJoCo not installed. Run: pip install mujoco")


# === MJCF Model for UR5-like arm ===

UR5_MJCF = """
<mujoco model="ur5_arm">
  <option gravity="0 0 -9.81" timestep="0.002"/>
  
  <default>
    <joint damping="0.5" armature="0.1"/>
    <geom contype="1" conaffinity="1" friction="1 0.5 0.5"/>
  </default>
  
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" 
             rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.2"/>
  </asset>
  
  <worldbody>
    <!-- Floor -->
    <geom name="floor" type="plane" size="1 1 0.1" material="grid"/>
    
    <!-- Table -->
    <geom name="table" type="box" pos="0.4 0 0.4" size="0.4 0.4 0.02" rgba="0.6 0.4 0.2 1"/>
    
    <!-- Robot Base -->
    <body name="base" pos="0 0 0.42">
      <geom name="base_geom" type="cylinder" size="0.06 0.03" rgba="0.3 0.3 0.3 1"/>
      <inertial pos="0 0 0" mass="1" diaginertia="0.01 0.01 0.01"/>
      
      <!-- Joint 1: Base rotation -->
      <body name="link1" pos="0 0 0.05">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom name="link1_geom" type="cylinder" size="0.05 0.1" rgba="0.5 0.5 0.5 1" 
              pos="0 0 0.1"/>
        <inertial pos="0 0 0.1" mass="0.5" diaginertia="0.005 0.005 0.003"/>
        
        <!-- Joint 2: Shoulder -->
        <body name="link2" pos="0 0 0.2">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
          <geom name="link2_geom" type="capsule" size="0.04" fromto="0 0 0 0.4 0 0" 
                rgba="0.6 0.3 0.2 1"/>
          <inertial pos="0.2 0 0" mass="0.4" diaginertia="0.004 0.004 0.002"/>
          
          <!-- Joint 3: Elbow -->
          <body name="link3" pos="0.4 0 0">
            <joint name="joint3" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
            <geom name="link3_geom" type="capsule" size="0.035" fromto="0 0 0 0.3 0 0" 
                  rgba="0.6 0.3 0.2 1"/>
            <inertial pos="0.15 0 0" mass="0.3" diaginertia="0.003 0.003 0.001"/>
            
            <!-- Joint 4: Wrist 1 -->
            <body name="link4" pos="0.3 0 0">
              <joint name="joint4" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
              <geom name="link4_geom" type="cylinder" size="0.03 0.04" rgba="0.4 0.4 0.4 1"/>
              <inertial pos="0 0 0" mass="0.2" diaginertia="0.001 0.001 0.001"/>
              
              <!-- Joint 5: Wrist 2 -->
              <body name="link5" pos="0 0 0.08">
                <joint name="joint5" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
                <geom name="link5_geom" type="cylinder" size="0.025 0.03" rgba="0.4 0.4 0.4 1"/>
                <inertial pos="0 0 0" mass="0.1" diaginertia="0.0005 0.0005 0.0005"/>
                
                <!-- Joint 6: Wrist 3 (tool) -->
                <body name="link6" pos="0 0 0.06">
                  <joint name="joint6" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
                  <geom name="gripper_base" type="box" size="0.03 0.02 0.01" rgba="0.2 0.2 0.2 1"/>
                  <inertial pos="0 0 0" mass="0.1" diaginertia="0.0003 0.0003 0.0003"/>
                  
                  <!-- End effector site -->
                  <site name="end_effector" pos="0 0 0.02" size="0.01" rgba="1 0 0 1"/>
                  
                  <!-- Gripper finger left -->
                  <body name="finger_left" pos="0 -0.02 0.02">
                    <joint name="finger_left_joint" type="slide" axis="0 1 0" range="-0.02 0.02"/>
                    <geom name="finger_left_geom" type="box" size="0.01 0.005 0.02" rgba="0.3 0.3 0.3 1"/>
                    <inertial pos="0 0 0" mass="0.02" diaginertia="0.00001 0.00001 0.00001"/>
                  </body>
                  
                  <!-- Gripper finger right -->
                  <body name="finger_right" pos="0 0.02 0.02">
                    <joint name="finger_right_joint" type="slide" axis="0 -1 0" range="-0.02 0.02"/>
                    <geom name="finger_right_geom" type="box" size="0.01 0.005 0.02" rgba="0.3 0.3 0.3 1"/>
                    <inertial pos="0 0 0" mass="0.02" diaginertia="0.00001 0.00001 0.00001"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
    
    <!-- Ball to pick up -->
    <body name="ball" pos="0.4 0.2 0.45">
      <freejoint name="ball_free"/>
      <geom name="ball_geom" type="sphere" size="0.025" rgba="1 0.2 0.2 1" mass="0.05"/>
    </body>
    
    <!-- Glass (container) -->
    <body name="glass" pos="0.4 -0.2 0.42">
      <geom name="glass_bottom" type="cylinder" size="0.04 0.005" rgba="0.8 0.8 1 0.5"/>
      <geom name="glass_wall" type="cylinder" size="0.04 0.05" pos="0 0 0.025" 
            rgba="0.8 0.8 1 0.3" contype="0"/>
    </body>
  </worldbody>
  
  <actuator>
    <motor name="motor1" joint="joint1" gear="50" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="motor2" joint="joint2" gear="50" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="motor3" joint="joint3" gear="50" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="motor4" joint="joint4" gear="30" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="motor5" joint="joint5" gear="30" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="motor6" joint="joint6" gear="20" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="gripper_left" joint="finger_left_joint" gear="5" ctrllimited="true" ctrlrange="-1 1"/>
    <motor name="gripper_right" joint="finger_right_joint" gear="5" ctrllimited="true" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


class MuJoCoArm:
    """Bras robotique contrôlé par MuJoCo.
    
    Interface entre LifeCore robotics et le simulateur physique MuJoCo.
    """
    
    def __init__(self, model_xml: str = UR5_MJCF, visualize: bool = True):
        if not MUJOCO_AVAILABLE:
            raise RuntimeError("MuJoCo not installed")
        
        # Charger le modèle
        self.model = mujoco.MjModel.from_xml_string(model_xml)
        self.data = mujoco.MjData(self.model)
        
        # Viewer optionnel
        self.visualize = visualize
        self.viewer = None
        
        # Joint names (6 DOF + 2 gripper)
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
        self.gripper_names = ["finger_left_joint", "finger_right_joint"]
        
        # Cache joint IDs
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) 
                          for name in self.joint_names]
        self.gripper_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) 
                            for name in self.gripper_names]
        
        # Actuator IDs
        self.actuator_ids = list(range(6))  # Motors 0-5 for joints
        self.gripper_actuator_ids = [6, 7]  # Motors 6-7 for gripper
        
        # Control gains
        self.kp = 50.0  # Position gain
        self.kd = 5.0   # Velocity gain
        
        # Target positions
        self.target_qpos = np.zeros(6)
        self.gripper_target = 0.0  # 0=open, 1=closed
        
        # Stats
        self.sim_time = 0.0
    
    def start_viewer(self):
        """Démarre le viewer MuJoCo."""
        if self.visualize and self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
    
    def close_viewer(self):
        """Ferme le viewer."""
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
    @property
    def joint_positions(self) -> np.ndarray:
        """Retourne les positions des 6 joints."""
        return np.array([self.data.qpos[i] for i in self.joint_ids])
    
    @property
    def joint_velocities(self) -> np.ndarray:
        """Retourne les vitesses des joints."""
        return np.array([self.data.qvel[i] for i in self.joint_ids])
    
    @property
    def end_effector_pos(self) -> np.ndarray:
        """Position de l'end-effector."""
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
        return self.data.site_xpos[site_id].copy()
    
    @property
    def ball_pos(self) -> np.ndarray:
        """Position de la balle."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
        return self.data.xpos[body_id].copy()
    
    @property
    def glass_pos(self) -> np.ndarray:
        """Position du verre."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "glass")
        return self.data.xpos[body_id].copy()
    
    def set_joint_targets(self, targets: np.ndarray):
        """Définit les positions cibles des joints."""
        self.target_qpos = np.clip(targets, -np.pi, np.pi)
    
    def set_gripper(self, closed: float):
        """Contrôle le gripper (0=ouvert, 1=fermé)."""
        self.gripper_target = np.clip(closed, 0.0, 1.0)
    
    def step(self, n_substeps: int = 10):
        """Avance la simulation de n_substeps."""
        for _ in range(n_substeps):
            # PD control pour les joints
            qpos = self.joint_positions
            qvel = self.joint_velocities
            
            error = self.target_qpos - qpos
            ctrl = self.kp * error - self.kd * qvel
            
            # Normaliser les contrôles [-1, 1]
            ctrl = np.clip(ctrl / 50.0, -1, 1)
            
            for i, actuator_id in enumerate(self.actuator_ids):
                self.data.ctrl[actuator_id] = ctrl[i]
            
            # Gripper control
            gripper_ctrl = (self.gripper_target - 0.5) * 2  # Map [0,1] to [-1,1]
            self.data.ctrl[self.gripper_actuator_ids[0]] = gripper_ctrl
            self.data.ctrl[self.gripper_actuator_ids[1]] = gripper_ctrl
            
            # Step simulation
            mujoco.mj_step(self.model, self.data)
            self.sim_time = self.data.time
        
        # Sync viewer
        if self.viewer:
            self.viewer.sync()
    
    def move_to_cartesian(self, target_pos: np.ndarray, max_iter: int = 100) -> bool:
        """Mouvement vers une position cartésienne (IK itératif simple)."""
        # IK par Jacobien (simplifié)
        for _ in range(max_iter):
            current_pos = self.end_effector_pos
            error = target_pos - current_pos
            
            if np.linalg.norm(error) < 0.01:
                return True
            
            # Jacobien numérique
            eps = 1e-4
            J = np.zeros((3, 6))
            for i in range(6):
                qpos_plus = self.joint_positions.copy()
                qpos_plus[i] += eps
                
                # Sauvegarder et modifier
                old_qpos = self.joint_positions.copy()
                self.set_joint_targets(qpos_plus)
                self.step(1)
                pos_plus = self.end_effector_pos
                
                # Restaurer
                self.set_joint_targets(old_qpos)
                self.step(1)
                
                J[:, i] = (pos_plus - current_pos) / eps
            
            # Pseudo-inverse avec damping
            damping = 0.1
            JTJ = J.T @ J + damping * np.eye(6)
            dq = np.linalg.solve(JTJ, J.T @ error)
            
            new_targets = self.joint_positions + 0.5 * dq
            self.set_joint_targets(new_targets)
            self.step(5)
        
        return False
    
    def run_demo(self, duration: float = 10.0):
        """Exécute une démo simple."""
        self.start_viewer()
        
        print("MuJoCo Demo - Ball in Glass")
        print("=" * 40)
        print(f"Ball: {self.ball_pos.round(3)}")
        print(f"Glass: {self.glass_pos.round(3)}")
        print(f"End effector: {self.end_effector_pos.round(3)}")
        print()
        
        dt = 0.02  # 50 Hz control
        steps = int(duration / dt)
        
        # Séquence: aller vers la balle, fermer, aller vers verre, ouvrir
        phase = 0
        phase_start = 0
        
        for step in range(steps):
            if not self.viewer.is_running():
                break
            
            t = step * dt
            
            # Phase 0: Aller vers la balle
            if phase == 0 and t > 0.5:
                print(f"Phase 1: Going to ball at {self.ball_pos.round(3)}")
                ball_above = self.ball_pos + np.array([0, 0, 0.08])
                self.move_to_cartesian(ball_above, max_iter=20)
                phase = 1
                phase_start = t
            
            # Phase 1: Descendre et fermer gripper
            elif phase == 1 and t - phase_start > 1.0:
                print("Phase 2: Closing gripper")
                self.set_gripper(1.0)
                phase = 2
                phase_start = t
            
            # Phase 2: Lever
            elif phase == 2 and t - phase_start > 1.0:
                print("Phase 3: Lifting")
                current = self.end_effector_pos
                self.move_to_cartesian(current + np.array([0, 0, 0.1]), max_iter=10)
                phase = 3
                phase_start = t
            
            # Phase 3: Aller vers verre
            elif phase == 3 and t - phase_start > 1.0:
                print(f"Phase 4: Going to glass at {self.glass_pos.round(3)}")
                glass_above = self.glass_pos + np.array([0, 0, 0.15])
                self.move_to_cartesian(glass_above, max_iter=20)
                phase = 4
                phase_start = t
            
            # Phase 4: Ouvrir gripper
            elif phase == 4 and t - phase_start > 1.5:
                print("Phase 5: Releasing")
                self.set_gripper(0.0)
                phase = 5
                phase_start = t
            
            # Phase 5: Done
            elif phase == 5 and t - phase_start > 1.0:
                print("Done!")
                break
            
            self.step(10)
            time.sleep(dt)
        
        print()
        print(f"Final ball position: {self.ball_pos.round(3)}")
        print(f"Glass position: {self.glass_pos.round(3)}")
        
        # Check if ball is in glass
        ball_xy = self.ball_pos[:2]
        glass_xy = self.glass_pos[:2]
        dist = np.linalg.norm(ball_xy - glass_xy)
        
        if dist < 0.04 and self.ball_pos[2] > self.glass_pos[2]:
            print("✅ Ball is in the glass!")
        else:
            print(f"❌ Ball not in glass (distance: {dist:.3f})")
        
        self.close_viewer()


def main():
    """Démonstration MuJoCo."""
    if not MUJOCO_AVAILABLE:
        print("MuJoCo not available. Install with:")
        print("  source venv311/bin/activate")
        print("  pip install mujoco")
        return
    
    arm = MuJoCoArm(visualize=True)
    arm.run_demo(duration=10.0)


if __name__ == "__main__":
    main()
