# Copyright 2025 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Proximal policy optimization training on a single device.

This version is refactored from the original to remove multi-GPU/multi-process
abstractions (like jax.pmap) but keeps all other functional components.

See: https://arxiv.org/pdf/1707.06347.pdf
"""

import functools
import time
from typing import Any, Callable, Mapping, Optional, Tuple, Union

from absl import logging
from brax import base
from brax import envs
from brax.training import acting
from brax.training import gradients
from brax.training import logger as metric_logger
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

# from


InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics


@flax.struct.dataclass
class TrainingState:
  """Contains training state for the learner."""

  optimizer_state: optax.OptState
  params: ppo_losses.PPONetworkParams
  normalizer_params: running_statistics.RunningStatisticsState
  env_steps: types.UInt64


def _strip_weak_type(tree):
  def f(leaf):
    leaf = jnp.asarray(leaf)
    return jnp.astype(leaf, leaf.dtype)

  return jax.tree_util.tree_map(f, tree)


def _maybe_wrap_env(
    env: envs.Env,
    wrap_env: bool,
    num_envs: int,
    episode_length: Optional[int],
    action_repeat: int,
    key_env: PRNGKey,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
):
  """Wraps the environment for training/eval if wrap_env is True."""
  if not wrap_env:
    return env
  if episode_length is None:
    raise ValueError('episode_length must be specified in ppo.train')
  
  v_randomization_fn = None
  if randomization_fn is not None:
    randomization_rng = jax.random.split(key_env, num_envs)
    v_randomization_fn = functools.partial(
        randomization_fn, rng=randomization_rng
    )
  
  if wrap_env_fn is not None:
    wrap_for_training = wrap_env_fn
  else:
    wrap_for_training = envs.training.wrap
    
  env = wrap_for_training(
      env,
      episode_length=episode_length,
      action_repeat=action_repeat,
      randomization_fn=v_randomization_fn,
  )
  return env


def _remove_pixels(
    obs: Union[jnp.ndarray, Mapping[str, jax.Array]],
) -> Union[jnp.ndarray, Mapping[str, jax.Array]]:
  """Removes pixel observations from the observation dict."""
  if not isinstance(obs, Mapping):
    return obs
  return {k: v for k, v in obs.items() if not k.startswith('pixels/')}


def train(
    environment: envs.Env,
    num_timesteps: int,
    wrap_env: bool = True,
    # environment wrapper
    num_envs: int = 1,
    episode_length: Optional[int] = None,
    action_repeat: int = 1,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    # ppo params
    learning_rate: float = 1e-4,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    unroll_length: int = 10,
    batch_size: int = 32,
    num_minibatches: int = 16,
    num_updates_per_batch: int = 2,
    num_resets_per_eval: int = 0,
    normalize_observations: bool = False,
    reward_scaling: float = 1.0,
    clipping_epsilon: float = 0.3,
    gae_lambda: float = 0.95,
    max_grad_norm: Optional[float] = None,
    normalize_advantage: bool = True,
    network_factory: types.NetworkFactory[
        ppo_networks.PPONetworks
    ] = ppo_networks.make_ppo_networks,
    seed: int = 0,
    # eval
    num_evals: int = 1,
    eval_env: Optional[envs.Env] = None,
    num_eval_envs: int = 128,
    deterministic_eval: bool = False,
    # training metrics
    log_training_metrics: bool = False,
    training_metrics_steps: Optional[int] = None,
    # callbacks
    progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    policy_params_fn: Callable[..., None] = lambda *args: None,
    # checkpointing
    save_checkpoint_path: Optional[str] = None,
    run_evals: bool = True,
    # custom ppo lf
    compute_custom_ppo_loss_fn = None
):
  """PPO training on a single device."""
  assert batch_size * num_minibatches % num_envs == 0
  xt = time.time()

  env_step_per_training_step = (
      batch_size * unroll_length * num_minibatches * action_repeat
  )
  num_evals_after_init = max(num_evals - 1, 1)
  num_training_steps_per_epoch = np.ceil(
      num_timesteps
      / (
          num_evals_after_init
          * env_step_per_training_step
          * max(num_resets_per_eval, 1)
      )
  ).astype(int)

  key = jax.random.PRNGKey(seed)
  key, key_env, eval_key, key_policy, key_value = jax.random.split(key, 5)

  env = _maybe_wrap_env(
      environment,
      wrap_env,
      num_envs,
      episode_length,
      action_repeat,
      key_env,
      wrap_env_fn,
      randomization_fn,
  )

  reset_fn = jax.jit(env.reset)
  
  key_envs = jax.random.split(key_env, num_envs)
  env_state = reset_fn(key_envs)
  obs_shape = jax.tree_util.tree_map(lambda x: x.shape[1:], env_state.obs)
  print(obs_shape)
  normalize = lambda x, y: x
  if normalize_observations:
    normalize = running_statistics.normalize

  ppo_network = network_factory(
      obs_shape, env.action_size, preprocess_observations_fn=normalize
  )
  make_policy = ppo_networks.make_inference_fn(ppo_network)

  optimizer = optax.adam(learning_rate=learning_rate)
  if max_grad_norm is not None:
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optimizer,
    )

  loss_fn = functools.partial(
      compute_custom_ppo_loss_fn,
      ppo_network=ppo_network,
      entropy_cost=entropy_cost,
      discounting=discounting,
      reward_scaling=reward_scaling,
      gae_lambda=gae_lambda,
      clipping_epsilon=clipping_epsilon,
      normalize_advantage=normalize_advantage,
  )

  gradient_update_fn = gradients.gradient_update_fn(
      loss_fn, optimizer, has_aux=True, pmap_axis_name=None
  )

  metrics_aggregator = metric_logger.EpisodeMetricsLogger(
      steps_between_logging=training_metrics_steps
      or env_step_per_training_step,
      progress_fn=progress_fn,
  )

  def minibatch_step(
      carry,
      data: types.Transition,
      normalizer_params: running_statistics.RunningStatisticsState,
  ):
    optimizer_state, params, key = carry
    key, key_loss = jax.random.split(key)
    (_, metrics), params, optimizer_state = gradient_update_fn(
        params,
        normalizer_params,
        data,
        key_loss,
        optimizer_state=optimizer_state,
    )
    return (optimizer_state, params, key), metrics

  def sgd_step(
      carry,
      unused_t,
      data: types.Transition,
      normalizer_params: running_statistics.RunningStatisticsState,
  ):
    optimizer_state, params, key = carry
    key, key_perm, key_grad = jax.random.split(key, 3)

    def convert_data(x: jnp.ndarray):
      x = jax.random.permutation(key_perm, x)
      x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
      return x

    shuffled_data = jax.tree_util.tree_map(convert_data, data)
    (optimizer_state, params, _), metrics = jax.lax.scan(
        functools.partial(minibatch_step, normalizer_params=normalizer_params),
        (optimizer_state, params, key_grad),
        shuffled_data,
        length=num_minibatches,
    )
    return (optimizer_state, params, key), metrics

  def training_step(
      carry: Tuple[TrainingState, envs.State, PRNGKey], unused_t
  ) -> Tuple[Tuple[TrainingState, envs.State, PRNGKey], Metrics]:
    training_state, state, key = carry
    key_sgd, key_generate_unroll, new_key = jax.random.split(key, 3)

    policy = make_policy((
        training_state.normalizer_params,
        training_state.params.policy,
        training_state.params.value,
    ))

    def f(carry, unused_t):
      current_state, current_key = carry
      current_key, next_key = jax.random.split(current_key)
      next_state, data = acting.generate_unroll(
          env,
          current_state,
          policy,
          current_key,
          unroll_length,
          extra_fields=('truncation', 'episode_metrics', 'episode_done'),
      )
      return (next_state, next_key), data

    (state, _), data = jax.lax.scan(
        f,
        (state, key_generate_unroll),
        (),
        length=batch_size * num_minibatches // num_envs,
    )
    data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
    data = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
    )
    
    if log_training_metrics:
      jax.debug.callback(
          metrics_aggregator.update_episode_metrics,
          data.extras['state_extras']['episode_metrics'],
          data.extras['state_extras']['episode_done'],
      )

    normalizer_params = running_statistics.update(
        training_state.normalizer_params,
        _remove_pixels(data.observation),
    )

    (optimizer_state, params, _), metrics = jax.lax.scan(
        functools.partial(
            sgd_step, data=data, normalizer_params=normalizer_params
        ),
        (training_state.optimizer_state, training_state.params, key_sgd),
        (),
        length=num_updates_per_batch,
    )

    new_training_state = TrainingState(
        optimizer_state=optimizer_state,
        params=params,
        normalizer_params=normalizer_params,
        env_steps=training_state.env_steps + env_step_per_training_step,
    )
    return (new_training_state, state, new_key), metrics

  def training_epoch(
      training_state: TrainingState, state: envs.State, key: PRNGKey
  ) -> Tuple[TrainingState, envs.State, Metrics]:
    (training_state, state, _), loss_metrics = jax.lax.scan(
        training_step,
        (training_state, state, key),
        (),
        length=num_training_steps_per_epoch,
    )
    loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
    return training_state, state, loss_metrics

  jit_training_epoch = jax.jit(training_epoch)

  init_params = ppo_losses.PPONetworkParams(
      policy=ppo_network.policy_network.init(key_policy),
      value=ppo_network.value_network.init(key_value),
  )
  obs_shape = jax.tree_util.tree_map(
      lambda x: specs.Array(x.shape[-1:], jnp.dtype('float32')), env_state.obs
  )

  training_state = TrainingState(
      optimizer_state=optimizer.init(init_params),
      params=init_params,
      normalizer_params=running_statistics.init_state(
          _remove_pixels(obs_shape)
      ),
      env_steps=0,
  )

  if num_timesteps == 0:
    return (make_policy, (training_state.normalizer_params, training_state.params.policy, training_state.params.value), {},)

  eval_env = _maybe_wrap_env(
      eval_env or environment,
      wrap_env,
      num_eval_envs,
      episode_length,
      action_repeat,
      eval_key,
      wrap_env_fn,
      randomization_fn,
  )
  evaluator = acting.Evaluator(
      eval_env,
      functools.partial(make_policy, deterministic=deterministic_eval),
      num_eval_envs=num_eval_envs,
      episode_length=episode_length,
      action_repeat=action_repeat,
      key=eval_key,
  )

  training_walltime = 0
  current_step = 0
  metrics = {}
  if num_evals > 1 and run_evals:
    metrics = evaluator.run_evaluation(
        (training_state.normalizer_params, training_state.params.policy, training_state.params.value),
        training_metrics={},
    )
    logging.info(metrics)
    progress_fn(0, metrics)

  params = (training_state.normalizer_params, training_state.params.policy, training_state.params.value)
  policy_params_fn(current_step, make_policy, params)

  for it in range(num_evals_after_init):
    logging.info('starting iteration %s %s', it, time.time() - xt)

    for _ in range(max(num_resets_per_eval, 1)):
      t = time.time()
      epoch_key, key = jax.random.split(key)
      training_state, env_state, training_metrics = jit_training_epoch(
          training_state, env_state, epoch_key
      )
      jax.tree_util.tree_map(lambda x: x.block_until_ready(), training_metrics)
      
      epoch_training_time = time.time() - t
      training_walltime += epoch_training_time
      sps = (num_training_steps_per_epoch * env_step_per_training_step) / epoch_training_time
      current_step = int(training_state.env_steps)
      
      metrics = {
          'training/sps': sps,
          'training/walltime': training_walltime,
          **{f'training/{name}': value for name, value in training_metrics.items()},
      }

      if num_resets_per_eval > 0:
        key, reset_key = jax.random.split(key)
        key_envs = jax.random.split(reset_key, num_envs)
        env_state = reset_fn(key_envs)

    params = (training_state.normalizer_params, training_state.params.policy, training_state.params.value)
    policy_params_fn(current_step, make_policy, params)

    if save_checkpoint_path is not None:
      ckpt_config = checkpoint.network_config(
          observation_size=obs_shape,
          action_size=env.action_size,
          normalize_observations=normalize_observations,
          network_factory=network_factory,
      )
      checkpoint.save(save_checkpoint_path, current_step, params, ckpt_config)

    if num_evals > 0:
      if run_evals:
        metrics = evaluator.run_evaluation(params, metrics)
      logging.info(metrics)
      progress_fn(current_step, metrics)

  total_steps = current_step
  if not total_steps >= num_timesteps:
    raise AssertionError(f'Total steps {total_steps} is less than `num_timesteps`= {num_timesteps}.')

  params = (training_state.normalizer_params, training_state.params.policy, training_state.params.value)
  logging.info('total steps: %s', total_steps)
  return (make_policy, params, metrics)