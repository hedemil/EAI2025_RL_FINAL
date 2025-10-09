#!/usr/bin/env python3
"""
Evaluate all saved PPO parameter files in `parameters/` using the same
evaluation function used during training (utils.evaluate_policy).

Intended to be launched from VS Code (no argparse). It will iterate over
`.npy` files in `parameters/` and display videos via mediapy.
"""
import os
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp  # noqa: F401
import numpy as np

from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.ppo import networks as ppo_networks

from utils import evaluate_policy


def build_policy(env, params_tuple: Tuple):
    """Constructs a deterministic inference policy from saved params."""
    reset_fn = jax.jit(env.reset)
    key = jax.random.PRNGKey(0)
    state = reset_fn(key)

    obs_shape = jax.tree_util.tree_map(
        lambda x: specs.Array(x.shape[-1:], jnp.dtype("float32")), state.obs
    )
    ppo_net = ppo_networks.make_ppo_networks(
        obs_shape,
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    make_policy = ppo_networks.make_inference_fn(ppo_net)
    # Returns (action, extras) to match utils.evaluate_policy expectations
    return make_policy(params_tuple, deterministic=True)


def run_all():
    # Ensure headless rendering for MuJoCo
    os.environ.setdefault("MUJOCO_GL", "egl")

    repo_root = Path(__file__).resolve().parents[1]
    params_dir = repo_root / "parameters"
    xml_path = repo_root / "environments" / "custom_env.xml"

    # Lazy import of environment after path setup
    from environments.custom_env import Joystick, default_config  # noqa: WPS433

    env_cfg = default_config()
    env = Joystick(xml_path=str(xml_path), config=env_cfg)
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    param_files = sorted(params_dir.glob("*.npy"))
    if not param_files:
        print(f"No .npy files found in {params_dir}")
        return

    for npy_path in param_files:
        print(f"\n=== Evaluating: {npy_path.name} ===")
        params_arr = np.load(npy_path, allow_pickle=True)
        if getattr(params_arr, "dtype", None) != object or len(params_arr) != 3:
            print(f"  Skipping {npy_path.name}: unexpected format {getattr(params_arr, 'dtype', None)} {getattr(params_arr, 'shape', None)}")
            continue
        normalizer_params, policy_params, value_params = params_arr
        params_tuple = (normalizer_params, policy_params, value_params)

        inference_fn = build_policy(env, params_tuple)
        # The evaluation function handles rendering and showing video
        evaluate_policy(env, jax.jit(inference_fn), jit_step, jit_reset, env_cfg, env)


if __name__ == "__main__":
    run_all()
