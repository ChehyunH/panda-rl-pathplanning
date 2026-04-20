"""
tests/test_env.py
=================
Unit tests for PandaReachEnv.

Run with:
    pytest tests/ -v
"""

import numpy as np
import pytest

# Guard: skip all tests if PyBullet is not installed
pybullet = pytest.importorskip("pybullet", reason="pybullet not installed")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.envs.panda_env import PandaReachEnv


@pytest.fixture(scope="module")
def env():
    """Create a single DIRECT-mode env for the test module."""
    e = PandaReachEnv(render=False, training=True, seed=0)
    yield e
    e.close()


# ── Spaces ────────────────────────────────────────────────────────────────────

def test_observation_space_shape(env):
    obs, _ = env.reset()
    assert obs.shape == (20,), f"Expected (20,), got {obs.shape}"


def test_action_space_shape(env):
    assert env.action_space.shape == (7,)


def test_obs_in_space(env):
    obs, _ = env.reset()
    assert env.observation_space.contains(obs), "Observation not in observation_space"


# ── Reset ─────────────────────────────────────────────────────────────────────

def test_reset_returns_obs_and_info(env):
    result = env.reset()
    assert isinstance(result, tuple) and len(result) == 2


def test_reset_clears_step_counter(env):
    env.reset()
    assert env.step_cnt == 0


def test_reset_clears_prev_action(env):
    env.reset()
    np.testing.assert_array_equal(env.prev_action, np.zeros(7))


def test_target_pos_within_workspace(env):
    for _ in range(5):
        env.reset()
        t = env.target_pos
        assert 0.25 <= t[0] <= 0.75, f"x={t[0]} out of expected workspace"
        assert -0.55 <= t[1] <= 0.55, f"y={t[1]} out of expected workspace"
        assert 0.15 <= t[2] <= 0.75, f"z={t[2]} out of expected workspace"


# ── Step ──────────────────────────────────────────────────────────────────────

def test_step_returns_five_tuple(env):
    env.reset()
    result = env.step(np.zeros(7, dtype=np.float32))
    assert len(result) == 5


def test_step_increments_counter(env):
    env.reset()
    env.step(np.zeros(7, dtype=np.float32))
    assert env.step_cnt == 1


def test_zero_action_does_not_crash(env):
    env.reset()
    obs, reward, terminated, truncated, info = env.step(np.zeros(7, dtype=np.float32))
    assert obs.shape == (20,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "distance" in info and "success" in info


def test_action_clipping(env):
    """Actions outside [-1, 1] should be clipped, not error."""
    env.reset()
    big_action = np.full(7, 100.0, dtype=np.float32)
    obs, _, _, _, _ = env.step(big_action)
    assert obs.shape == (20,)


def test_truncation_at_max_steps(env):
    env.reset()
    done = False
    step = 0
    while not done:
        _, _, terminated, truncated, _ = env.step(np.zeros(7, dtype=np.float32))
        done = terminated or truncated
        step += 1
        if step > env.max_steps + 5:
            break
    assert step <= env.max_steps + 1, "Episode did not truncate at max_steps"


# ── Reward ─────────────────────────────────────────────────────────────────────

def test_reward_is_scalar(env):
    env.reset()
    _, reward, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
    assert isinstance(reward, (int, float))


def test_eval_mode_no_shaping(env):
    """In training=False mode, _compute_reward should skip energy penalties."""
    eval_env = PandaReachEnv(render=False, training=False, seed=1)
    eval_env.reset()
    _, r1, _, _, _ = eval_env.step(np.zeros(7, dtype=np.float32))
    eval_env.close()

    train_env = PandaReachEnv(render=False, training=True, seed=1)
    train_env.reset()
    _, r2, _, _, _ = train_env.step(np.zeros(7, dtype=np.float32))
    train_env.close()

    # Training reward includes penalties → generally lower than eval reward
    # (both start at same state, so distance is the same)
    assert r2 <= r1, "Training reward should be ≤ eval reward due to penalties"


# ── Reproducibility ───────────────────────────────────────────────────────────

def test_same_seed_same_target():
    targets = []
    for _ in range(2):
        e = PandaReachEnv(render=False, training=False, seed=7)
        e.reset(seed=7)
        targets.append(e.target_pos.copy())
        e.close()
    np.testing.assert_allclose(targets[0], targets[1], atol=1e-6)
