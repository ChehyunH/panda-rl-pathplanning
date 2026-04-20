"""
train.py — SAC training for Franka Panda Reach task
=====================================================
Usage:
    python scripts/train.py --config configs/sac_default.yaml
    python scripts/train.py --timesteps 5000000 --n-envs 8
"""

import argparse
import os
import yaml
from datetime import datetime

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.envs.panda_env import PandaReachEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC on PandaReachEnv")
    parser.add_argument("--config",     type=str, default=None,       help="Path to YAML config")
    parser.add_argument("--timesteps",  type=int, default=10_000_000, help="Total training timesteps")
    parser.add_argument("--n-envs",     type=int, default=8,          help="Number of parallel envs")
    parser.add_argument("--seed",       type=int, default=42,         help="Random seed")
    parser.add_argument("--model-name", type=str, default=None,       help="Output model filename")
    parser.add_argument("--log-dir",    type=str, default="results/logs")
    parser.add_argument("--model-dir",  type=str, default="results/models")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_env(seed: int, training: bool = True):
    def _init():
        env = PandaReachEnv(render=False, training=training, seed=seed)
        return env
    return _init


def main():
    args = parse_args()

    # Load YAML config if provided (overrides CLI defaults)
    cfg = {}
    if args.config and os.path.exists(args.config):
        cfg = load_config(args.config)

    timesteps  = cfg.get("timesteps",  args.timesteps)
    n_envs     = cfg.get("n_envs",     args.n_envs)
    seed       = cfg.get("seed",       args.seed)
    lr         = cfg.get("lr",         2e-4)
    batch_size = cfg.get("batch_size", 512)

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model_name or cfg.get("model_name", f"sac_panda_{timestamp}")

    os.makedirs(args.log_dir,   exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    print(f"[Train] timesteps={timesteps:,} | n_envs={n_envs} | seed={seed} | lr={lr}")

    # Training envs
    train_env = make_vec_env(
        PandaReachEnv,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={"render": False, "training": True},
    )
    train_env = VecMonitor(train_env, filename=os.path.join(args.log_dir, model_name))

    # Eval env (single, no reward shaping, deterministic)
    eval_env = PandaReachEnv(render=False, training=False, seed=seed + 999)

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1),
        save_path=os.path.join(args.model_dir, "checkpoints", model_name),
        name_prefix="ckpt",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(args.model_dir, "best"),
        log_path=args.log_dir,
        eval_freq=max(50_000 // n_envs, 1),
        n_eval_episodes=20,
        deterministic=True,
        verbose=1,
    )

    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=seed,
        learning_rate=lr,
        batch_size=batch_size,
        tensorboard_log=args.log_dir,
    )

    print(f"[Train] Starting training → {model_name}")
    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_cb, eval_cb],
        tb_log_name=model_name,
    )

    save_path = os.path.join(args.model_dir, model_name)
    model.save(save_path)
    print(f"[Train] Model saved → {save_path}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
