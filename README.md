# 🦾 Panda RL Path Planning

> **Dynamic stability-aware path optimization for the 7-DOF Franka Panda robot arm using Soft Actor-Critic (SAC) reinforcement learning.**

---

## 📋 Project Background

During a robotic grasping demo, two critical problems appeared:

| Problem | Symptom |
|---|---|
| **Unclear target approach** | End-effector (EE) position was ambiguous despite object detection |
| **Dynamic instability** | All joints snapped aggressively toward the goal → base frame vibration → EE error |

This project solves both by learning a **smooth, energy-efficient** motion policy that reaches arbitrary targets within the robot's workspace with **≥ 95% accuracy at < 5 mm error**.

---

## 🗂️ Project Structure

```
panda-rl-pathplanning/
├── src/
│   └── envs/
│       └── panda_env.py        # Unified train/eval Gymnasium environment
├── scripts/
│   ├── train.py                # SAC training entry point
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
git clone https://github.com/<your-username>/panda-rl-pathplanning.git
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

## 🚀 Usage

### Train a new model
```bash
# With default config
python scripts/train.py

# Override specific parameters
python scripts/train.py --timesteps 5000000 --n-envs 8 --seed 42

# Use a custom YAML config
python scripts/train.py --config configs/sac_default.yaml
```

### Evaluate a trained model
```bash
# Headless evaluation (50 episodes, prints summary)
python scripts/evaluate.py --model results/models/sac_panda.zip

# With GUI and fewer episodes
python scripts/evaluate.py --model results/models/sac_panda.zip --render --episodes 20

# Save results to JSON
python scripts/evaluate.py --model results/models/sac_panda.zip --out results/logs/eval.json
```

### Monitor training with TensorBoard
```bash
tensorboard --logdir results/logs/
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

### Reward Function
```
reward = - distance × 20.0                        # (1) Distance penalty
       + (0.02 - distance) × 200.0  [if < 2 cm]   # (2) Precision bonus
       - 0.01 × mean(torques²)                     # (3) Energy penalty  ← training only
       - 0.10 × mean((aₜ - aₜ₋₁)²)               # (4) Jerk penalty     ← training only
       + 150.0                       [on success]  # (5) Success bonus
       - 10.0                        [on collision] # (6) Collision penalty
```

> **Key design note:** Penalties (3) and (4) are applied **during training only**.  
> Evaluation uses a clean distance-based metric for fair comparison.

### Termination Conditions
- ✅ **Success:** `distance < 5 mm`
- 💥 **Collision:** self-collision detected
- ⏱️ **Truncation:** `step_count >= max_steps (200)`

---

## 📊 Results

| Metric | Value |
|---|---|
| Success rate (< 5 mm) | **95%** |
| Within 10 mm | ~99% |
| Training steps | 10M (SAC, 14 parallel envs) |

### Qualitative comparison

| Without dynamic stability reward | With dynamic stability reward |
|---|---|
| Joints snap aggressively to target | Smooth, energy-efficient motion |
| Base frame visibly vibrates | Stable throughout trajectory |

> Demo videos: see [`docs/media/`](docs/media/)

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
| Parallel envs | `8–14` |

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
