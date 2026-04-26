"""
evaluate_now.py — 바로 실행 가능한 평가 스크립트
==================================================
실행: python evaluate_now.py
모델 파일 경로를 MODEL_PATH에 맞게 수정하세요.
"""

import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data as pd
import numpy as np
import time
import os
from stable_baselines3 import SAC

# ==============================================================================
# 설정
# ==============================================================================
MODEL_PATH     = "models/sac_panda.zip"  # ← 모델 경로 (필요시 수정)
NUM_EPISODES   = 50
RENDER         = True
STEP_DELAY     = 0.03
ACTION_SCALE   = 0.03
MAX_STEPS      = 200
SUCCESS_THRESH = 0.005

# ==============================================================================
# 환경 (train_now.py와 동일 + 버그 수정)
# ==============================================================================
class PandaReachEnv(gym.Env):

    JOINT_INDICES = [0, 1, 2, 3, 4, 5, 6]
    EE_LINK_INDEX = 11
    JOINT_LL = np.array([-2.89, -1.76, -2.89, -3.07, -2.89, -0.01, -2.89])
    JOINT_UL = np.array([ 2.89,  1.76,  2.89, -0.06,  2.89,  3.75,  2.89])
    SIM_STEP   = 1.0 / 240.0
    CONTROL_DT = 0.05

    def __init__(self, render=False):
        super().__init__()
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
                # ✅ 자세 원복 후 리턴
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
        distance = float(np.linalg.norm(self.target_pos - obs[14:17]))
        terminated = distance < SUCCESS_THRESH
        truncated  = self.step_cnt >= MAX_STEPS
        self.prev_action = action.copy()

        return obs, 0.0, terminated, truncated, {
            "distance": distance,
            "success":  terminated,
        }

    def close(self):
        if p.isConnected():
            p.disconnect()


# ==============================================================================
# 평가
# ==============================================================================
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 모델 없음: {MODEL_PATH}")
        print("   MODEL_PATH를 맞게 수정하세요.")
        exit()

    print(f"📦 모델 로드: {MODEL_PATH}")
    model = SAC.load(MODEL_PATH)
    env   = PandaReachEnv(render=RENDER)

    successes = []
    distances = []

    for ep in range(NUM_EPISODES):
        obs, _ = env.reset()
        done   = False
        steps  = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            if RENDER:
                time.sleep(STEP_DELAY)

        successes.append(info["success"])
        distances.append(info["distance"])

        status = "✅ 성공" if info["success"] else "❌ 실패"
        print(f"  [{ep+1:>2}/{NUM_EPISODES}] {status} | 오차: {info['distance']*1000:.2f}mm | 스텝: {steps}")

        if RENDER:
            time.sleep(1.0)

    print("\n" + "─" * 45)
    print(f"📊 성공률     : {np.mean(successes)*100:.1f}%")
    print(f"   평균 오차  : {np.mean(distances)*1000:.2f}mm")
    print(f"   최대 오차  : {np.max(distances)*1000:.2f}mm")
    print(f"   10mm 이내  : {np.mean([d < 0.01 for d in distances])*100:.1f}%")
    print("─" * 45)

    env.close()
