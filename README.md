# 🦾 Panda RL Path Planning

> **Dynamic stability-aware path optimization for the 7-DOF Franka Panda robot arm using Soft Actor-Critic (SAC) reinforcement learning.**

---

## 📋 Project Background

During a robotic grasping demo, two critical problems appeared:

| Problem | Symptom |
|---|---|
| **Unclear target approach** | End-effector (EE) position was ambiguous despite object detection |
| **Dynamic instability** | All joints snapped aggressively toward the goal → base frame vibration → EE error |

This project solves both by learning a **smooth, energy-efficient** motion policy that reaches arbitrary targets within the robot's workspace with **94% accuracy at < 5 mm error**.

---

## 🗂️ Project Structure

```
panda-rl-pathplanning/
├── train_now.py                # 바로 실행 가능한 학습 스크립트
├── evaluate_now.py             # 바로 실행 가능한 평가 스크립트
├── src/
│   └── envs/
│       └── panda_env.py        # Unified train/eval Gymnasium environment
├── scripts/
│   ├── train.py                # SAC training entry point (with CLI args)
│   └── evaluate.py             # Model evaluation with metrics
├── tests/
│   └── test_env.py             # Pytest unit tests for the environment
├── configs/
│   └── sac_default.yaml        # Hyper-parameter config
├── results/
│   ├── models/                 # Saved .zip model files
│   ├── logs/                   # TensorBoard & monitor logs
│   └── plots/                  # Generated figures
└── docs/
    └── media/                  # Demo videos, diagrams
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/ChehyunH/panda-rl-pathplanning.git
cd panda-rl-pathplanning
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> **Windows users:** PyBullet requires the MSVC C++ build tools.  
> Install from [Visual Studio](https://visualstudio.microsoft.com/downloads/) and select **"Desktop development with C++"**.

---

## 🚀 Quick Start

### Train
```bash
python train_now.py
```

### Evaluate
```bash
python evaluate_now.py
```

---

## 🚀 Advanced Usage

### Train with CLI options
```bash
python scripts/train.py --timesteps 10000000 --n-envs 6 --seed 42
python scripts/train.py --config configs/sac_default.yaml
```

### Evaluate with options
```bash
python scripts/evaluate.py --model results/models/sac_panda.zip --render --episodes 50
python scripts/evaluate.py --model results/models/sac_panda.zip --out results/logs/eval.json
```

### Monitor training with TensorBoard
```bash
tensorboard --logdir logs/
```

### Run tests
```bash
pytest tests/ -v
```

---

## 🧠 Environment Design

### Observation Space (20,)
| Index | Description |
|---|---|
| `[0:7]`   | Joint positions (rad) |
| `[7:14]`  | Joint velocities (rad/s) |
| `[14:17]` | End-effector position (m) |
| `[17:20]` | Relative position to target (m) |

### Action Space (7,)
Normalised delta joint positions `[-1.0, 1.0]`, scaled by `action_scale=0.03`.

> The small action scale (0.03 rad/step) is intentional — it produces smooth, stable motion at the cost of speed. This directly addresses the dynamic instability problem from the original demo.

### Reward Function
```
reward = - distance × 20.0                        # (1) Distance penalty
       + (0.02 - distance) × 200.0  [if < 2 cm]   # (2) Precision bonus
       - 0.01 × mean(torques²)                     # (3) Energy penalty  ← training only
       - 0.10 × mean((aₜ - aₜ₋₁)²)               # (4) Jerk penalty     ← training only
       + 150.0                       [on success]  # (5) Success bonus
       - 10.0                        [on collision] # (6) Collision penalty
```

> Penalties (3) and (4) are applied **during training only**.  
> Evaluation uses a clean distance-based metric for fair comparison.

### Termination Conditions
- ✅ **Success:** `distance < 5 mm`
- 💥 **Collision:** self-collision detected
- ⏱️ **Truncation:** `step_count >= max_steps (200)`

---

## 📊 Results

Evaluated over **50 episodes** with fully randomised targets across the workspace.  
Both training and evaluation use the bug-fixed environment (see Bug Fix section below).

| Metric | Value |
|---|---|
| **Success rate (< 5 mm)** | **94.0%** |
| **Within 10 mm** | **94.0%** |
| **Mean distance** | 7.06 mm |
| **Max distance** | 100.74 mm (workspace boundary case) |
| **Mean steps to success** | ~46 steps |
| Training steps | 10M |
| Parallel envs | 6 (i5-11400F) |
| Training time | ~36 hours |

### Failure analysis
| Episode | Error | Cause |
|---|---|---|
| 8 | 18 mm | Near workspace boundary |
| 12 | 100 mm | Extreme edge of reachable space |
| 42 | 34 mm | Near workspace boundary |

All 3 failures occurred at the boundary of the robot's reachable workspace, not in the central working area.

### Qualitative comparison

| Without dynamic stability reward | With dynamic stability reward |
|---|---|
| Joints snap aggressively to target | Smooth, incremental motion (~46 steps avg) |
| Base frame visibly vibrates | Stable throughout trajectory |
| High torque spikes | Energy-efficient movement |

> Demo videos: see [`docs/media/`](docs/media/)

---

## 🐛 Bug Fix: IK Pose Contamination

A critical bug was discovered and fixed in the target sampling process.

**The problem:** During IK validation, the robot was physically moved to the target position to verify reachability. The robot was never restored to its original pose afterward — so episodes started with the robot already near the target.

**The symptom:** The buggy model appeared to succeed in 1–2 steps, giving a falsely inflated success rate. The corrected evaluation shows the real performance requires ~46 steps per episode.

**The fix:**
```python
# Save pose before IK validation
original_q = [p.getJointState(self.robot, i)[0] for i in self.JOINT_INDICES]

# ... IK validation ...

if np.linalg.norm(cand - actual) < 0.01:
    # Restore pose after validation ✅
    for i, idx in enumerate(self.JOINT_INDICES):
        p.resetJointState(self.robot, idx, original_q[i])
    return cand
```

---

## 🔬 Algorithm

**Soft Actor-Critic (SAC)** was chosen for:
- Continuous action space support
- Maximum entropy framework → natural exploration of the redundant 7-DOF solution space
- Off-policy learning with replay buffer → sample efficiency

| Hyper-parameter | Value |
|---|---|
| Learning rate | `2e-4` |
| Batch size | `512` |
| Buffer size | `1,000,000` |
| Entropy coefficient | `auto` |
| Parallel envs | `6` |
| Action scale | `0.03` rad/step |

---

## 🔭 Future Work

Next project: **Dynamic stability verification** via external perturbation testing.

Planned experiments:
- **Impulse** input → settling time, overshoot analysis
- **Unit step** input → response speed
- **Ramp** input → trajectory tracking error
- **Harmonic** input → Bode plot (frequency response of EE displacement)

---

## 📄 License

MIT License — see [`LICENSE`](LICENSE) for details.
