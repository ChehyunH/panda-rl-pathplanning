"""
train_now.py — 바로 실행 가능한 SAC 학습 스크립트
====================================================
실행: python train_now.py
"""

import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data as pd
import numpy as np
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

# ==============================================================================
# 설정 (여기만 바꾸면 됨)
# ==============================================================================
NUM_CPU        = 6           # i5-11400F 기준 최적값
TOTAL_STEPS    = 10_000_000  # 총 학습 횟수
MODEL_NAME     = "sac_panda" # 저장 파일명
LEARNING_RATE  = 2e-4
BATCH_SIZE     = 512
ACTION_SCALE   = 0.03
MAX_STEPS      = 200
SUCCESS_THRESH = 0.005       # 5mm

# ==============================================================================
# 환경
# ==============================================================================
class PandaReachEnv(gym.Env):

    JOINT_INDICES = [0, 1, 2, 3, 4, 5, 6]
    EE_LINK_INDEX = 11
    JOINT_LL = np.array([-2.89, -1.76, -2.89, -3.07, -2.89, -0.01, -2.89])
    JOINT_UL = np.array([ 2.89,  1.76,  2.89, -0.06,  2.89,  3.75,  2.89])
    SIM_STEP   = 1.0 / 240.0
    CONTROL_DT = 0.05

    def __init__(self, render=False, training=True):
        super().__init__()
        self.training    = training
        self.render_mode = "human" if render else None
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

        self.robot       = None
        self.target_pos  = None
        self.step_cnt    = 0
        self.prev_action = np.zeros(7, dtype=np.float32)

        if render:
            p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.resetDebugVisualizerCamera(1.2, 45, -30, [0.5, 0, 0.3])
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pd.getDataPath())

    def _get_obs(self):
        js  = p.getJointStates(self.robot, self.JOINT_INDICES)
        q   = np.array([s[0] for s in js], dtype=np.float32)
        dq  = np.array([s[1] for s in js], dtype=np.float32)
        ee  = np.array(p.getLinkState(self.robot, self.EE_LINK_INDEX)[0], dtype=np.float32)
        rel = (self.target_pos - ee).astype(np.float32)
        return np.concatenate([q, dq, ee, rel])

    def _sample_target(self):
        # 현재 자세 저장 (IK 검증이 자세를 오염시키는 버그 수정)
        original_q = [p.getJointState(self.robot, i)[0] for i in self.JOINT_INDICES]

        for _ in range(100):
            cand = np.array([
                np.random.uniform(0.3, 0.7),
                np.random.uniform(-0.5, 0.5),
                np.random.uniform(0.2, 0.7),
            ])
            ik = p.calculateInverseKinematics(self.robot, self.EE_LINK_INDEX, cand)
            for i, idx in enumerate(self.JOINT_INDICES):
                p.resetJointState(self.robot, idx, ik[i])
            p.stepSimulation()
            actual = np.array(p.getLinkState(self.robot, self.EE_LINK_INDEX)[0])

            if np.linalg.norm(cand - actual) < 0.01:
                # ✅ 자세 원복 후 리턴 (핵심 버그 수정)
                for i, idx in enumerate(self.JOINT_INDICES):
                    p.resetJointState(self.robot, idx, original_q[i])
                return cand

        # 100번 실패해도 원복
        for i, idx in enumerate(self.JOINT_INDICES):
            p.resetJointState(self.robot, idx, original_q[i])
        return np.array([0.5, 0.0, 0.4], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        self.robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

        for i, idx in enumerate(self.JOINT_INDICES):
            p.resetJointState(self.robot, idx,
                np.random.uniform(self.JOINT_LL[i] * 0.1, self.JOINT_UL[i] * 0.1))

        self.target_pos  = self._sample_target()
        self.step_cnt    = 0
        self.prev_action = np.zeros(7, dtype=np.float32)

        if self.render_mode == "human":
            v = p.createVisualShape(p.GEOM_SPHERE, radius=SUCCESS_THRESH, rgbaColor=[0, 1, 0, 0.9])
            p.createMultiBody(baseVisualShapeIndex=v, basePosition=self.target_pos)

        return self._get_obs(), {}

    def step(self, action):
        self.step_cnt += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        current_q = np.array([p.getJointState(self.robot, i)[0] for i in self.JOINT_INDICES])
        target_q  = np.clip(current_q + action * ACTION_SCALE, self.JOINT_LL, self.JOINT_UL)
        p.setJointMotorControlArray(self.robot, self.JOINT_INDICES, p.POSITION_CONTROL,
                                    targetPositions=target_q, forces=[500] * 7)

        for _ in range(int(self.CONTROL_DT / self.SIM_STEP)):
            p.stepSimulation()

        obs      = self._get_obs()
        ee_pos   = obs[14:17]
        distance = float(np.linalg.norm(self.target_pos - ee_pos))
        torques  = np.array([p.getJointState(self.robot, i)[3] for i in self.JOINT_INDICES])

        # 보상
        reward = -distance * 20.0
        if distance < 0.02:
            reward += (0.02 - distance) * 200.0
        if self.training:
            reward -= 0.01 * float(np.mean(np.square(torques)))
            reward -= 0.10 * float(np.mean(np.square(action - self.prev_action)))

        # 종료 조건
        contacts  = p.getContactPoints(self.robot, self.robot)
        collision = len(contacts) > 0
        if collision:
            reward    -= 10.0
            terminated = True
        elif distance < SUCCESS_THRESH:
            reward    += 150.0
            terminated = True
        else:
            terminated = False

        truncated        = self.step_cnt >= MAX_STEPS
        self.prev_action = action.copy()

        return obs, reward, terminated, truncated, {
            "distance": distance,
            "success":  distance < SUCCESS_THRESH,
        }

    def close(self):
        if p.isConnected():
            p.disconnect()


# ==============================================================================
# 학습
# ==============================================================================
if __name__ == "__main__":
    os.makedirs("logs",   exist_ok=True)
    os.makedirs("models", exist_ok=True)

    print(f"🚀 학습 시작 | CPU: {NUM_CPU}개 | 총 스텝: {TOTAL_STEPS:,}")

    train_env = make_vec_env(
        PandaReachEnv,
        n_envs=NUM_CPU,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"render": False, "training": True},
    )
    train_env = VecMonitor(train_env, filename="logs/monitor")

    checkpoint_cb = CheckpointCallback(
        save_freq=max(200_000 // NUM_CPU, 1),
        save_path="models/checkpoints/",
        name_prefix="ckpt",
        verbose=1,
    )

    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        tensorboard_log="logs/",
    )

    model.learn(total_timesteps=TOTAL_STEPS, callback=checkpoint_cb, tb_log_name=MODEL_NAME)
    model.save(f"models/{MODEL_NAME}")
    print(f"✅ 저장 완료 → models/{MODEL_NAME}.zip")

    train_env.close()
