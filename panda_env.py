"""
Franka Panda Reach Environment
================================
Unified training/testing environment for the 7-DOF Franka Panda robot arm.
Supports dynamic stability rewards, energy minimization, and collision detection.
"""

import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data as pd
import numpy as np
from typing import Optional, Tuple, Dict, Any


class PandaReachEnv(gym.Env):
    """
    Franka Panda 7-DOF robot arm reach task.

    Observation space (20,):
        - Joint positions       [0:7]
        - Joint velocities      [7:14]
        - End-effector position [14:17]
        - Relative target pos   [17:20]

    Action space (7,):
        - Delta joint positions [-1.0, 1.0] per joint (scaled by action_scale)

    Reward:
        - Distance penalty (scaled)
        - Precision bonus (when close)
        - Torque / energy penalty
        - Jerk penalty (action smoothness)
        - Collision penalty
        - Success bonus

    Args:
        render (bool): Enable PyBullet GUI rendering.
        training (bool): If False, disables reward shaping (for clean eval).
        action_scale (float): Scales action magnitude for fine control.
        max_steps (int): Max steps per episode before truncation.
        success_threshold (float): Distance (m) considered a success.
        seed (Optional[int]): Random seed for reproducibility.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    # ── Robot constants ──────────────────────────────────────────────────────
    JOINT_INDICES  = [0, 1, 2, 3, 4, 5, 6]
    EE_LINK_INDEX  = 11
    JOINT_LL       = np.array([-2.89, -1.76, -2.89, -3.07, -2.89, -0.01, -2.89])
    JOINT_UL       = np.array([ 2.89,  1.76,  2.89, -0.06,  2.89,  3.75,  2.89])
    MAX_FORCE      = 500.0

    # ── Simulation constants ─────────────────────────────────────────────────
    SIM_STEP       = 1.0 / 240.0
    CONTROL_DT     = 0.05  # seconds per RL step

    def __init__(
        self,
        render: bool = False,
        training: bool = True,
        action_scale: float = 0.03,
        max_steps: int = 200,
        success_threshold: float = 0.005,
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.render_mode       = "human" if render else None
        self.training          = training
        self.action_scale      = action_scale
        self.max_steps         = max_steps
        self.success_threshold = success_threshold

        # Gym spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
        )

        # Episode state (initialised in reset)
        self.robot:      Optional[int]       = None
        self.target_pos: Optional[np.ndarray] = None
        self.step_cnt:   int                 = 0
        self.prev_action: np.ndarray         = np.zeros(7, dtype=np.float32)
        self.prev_velocity: np.ndarray       = np.zeros(7, dtype=np.float32)

        # PyBullet connection
        if render:
            self._physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.2,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0.5, 0, 0.3],
            )
        else:
            self._physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pd.getDataPath())

        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        joint_states = p.getJointStates(self.robot, self.JOINT_INDICES)
        q   = np.array([s[0] for s in joint_states], dtype=np.float32)
        dq  = np.array([s[1] for s in joint_states], dtype=np.float32)
        ee_pos = np.array(p.getLinkState(self.robot, self.EE_LINK_INDEX)[0], dtype=np.float32)
        rel_pos = (self.target_pos - ee_pos).astype(np.float32)
        return np.concatenate([q, dq, ee_pos, rel_pos])

    def _get_torques(self) -> np.ndarray:
        joint_states = p.getJointStates(self.robot, self.JOINT_INDICES)
        return np.array([s[3] for s in joint_states], dtype=np.float32)

    def _sample_valid_target(self) -> np.ndarray:
        """
        Sample a reachable Cartesian target via IK validation.
        Raises a warning and falls back to a safe default if IK fails.
        """
        target_ranges = {
            "x": (0.3, 0.7),
            "y": (-0.5, 0.5),
            "z": (0.2, 0.7),
        }
        rng = self.np_random if hasattr(self, "np_random") else np.random

        for attempt in range(100):
            candidate = np.array([
                rng.uniform(*target_ranges["x"]),
                rng.uniform(*target_ranges["y"]),
                rng.uniform(*target_ranges["z"]),
            ])
            ik_solution = p.calculateInverseKinematics(
                self.robot, self.EE_LINK_INDEX, candidate
            )
            for i, idx in enumerate(self.JOINT_INDICES):
                p.resetJointState(self.robot, idx, ik_solution[i])
            p.stepSimulation()
            actual = np.array(p.getLinkState(self.robot, self.EE_LINK_INDEX)[0])
            if np.linalg.norm(candidate - actual) < 0.01:
                return candidate

        import warnings
        warnings.warn(
            "IK failed after 100 attempts — using fallback target [0.5, 0.0, 0.4]. "
            "Check workspace limits.",
            RuntimeWarning,
        )
        return np.array([0.5, 0.0, 0.4], dtype=np.float32)

    def _visualise_target(self, pos: np.ndarray) -> None:
        v_id = p.createVisualShape(
            p.GEOM_SPHERE, radius=self.success_threshold, rgbaColor=[0, 1, 0, 0.9]
        )
        p.createMultiBody(baseVisualShapeIndex=v_id, basePosition=pos)

    # ── Gym API ──────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

        # Randomise start pose (small perturbation around zero)
        for i, idx in enumerate(self.JOINT_INDICES):
            init_q = self.np_random.uniform(
                self.JOINT_LL[i] * 0.1, self.JOINT_UL[i] * 0.1
            )
            p.resetJointState(self.robot, idx, init_q)

        self.target_pos  = self._sample_valid_target()
        self.step_cnt    = 0
        self.prev_action = np.zeros(7, dtype=np.float32)
        self.prev_velocity = np.zeros(7, dtype=np.float32)

        if self.render_mode == "human":
            self._visualise_target(self.target_pos)

        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.step_cnt += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Delta position control
        current_q = np.array(
            [p.getJointState(self.robot, i)[0] for i in self.JOINT_INDICES]
        )
        target_q = np.clip(
            current_q + action * self.action_scale, self.JOINT_LL, self.JOINT_UL
        )
        p.setJointMotorControlArray(
            self.robot,
            self.JOINT_INDICES,
            p.POSITION_CONTROL,
            targetPositions=target_q,
            forces=[self.MAX_FORCE] * 7,
        )

        for _ in range(int(self.CONTROL_DT / self.SIM_STEP)):
            p.stepSimulation()

        obs      = self._get_obs()
        ee_pos   = obs[14:17]
        distance = float(np.linalg.norm(self.target_pos - ee_pos))
        torques  = self._get_torques()

        # ── Reward ──────────────────────────────────────────────────────────
        reward = self._compute_reward(distance, torques, action)

        # ── Termination ─────────────────────────────────────────────────────
        contacts = p.getContactPoints(self.robot, self.robot)
        collision = len(contacts) > 0

        if collision:
            reward    -= 10.0
            terminated = True
        elif distance < self.success_threshold:
            reward    += 150.0
            terminated = True
        else:
            terminated = False

        truncated = self.step_cnt >= self.max_steps

        # Update history
        self.prev_action   = action.copy()
        self.prev_velocity = obs[7:14].copy()

        info = {
            "distance":  distance,
            "success":   distance < self.success_threshold,
            "collision": collision,
        }
        return obs, reward, terminated, truncated, info

    def _compute_reward(
        self,
        distance: float,
        torques: np.ndarray,
        action: np.ndarray,
    ) -> float:
        # (1) Distance penalty
        reward = -distance * 20.0

        # (2) Precision bonus
        if distance < 0.02:
            reward += (0.02 - distance) * 200.0

        if not self.training:
            return reward  # clean eval: no shaping penalties

        # (3) Energy / torque penalty
        reward -= 0.01 * float(np.mean(np.square(torques)))

        # (4) Jerk penalty (action smoothness)
        reward -= 0.1 * float(np.mean(np.square(action - self.prev_action)))

        return reward

    def close(self) -> None:
        if p.isConnected():
            p.disconnect()
