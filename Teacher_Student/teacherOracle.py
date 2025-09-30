from typing import Tuple, Any, Dict
import jax
import jax.numpy as jnp

class TeacherOracle:
    def __init__(self,
                 make_policy,        # ppo_networks.make_inference_fn(...)
                 normalizer_params,  # running_stats for teacher obs
                 policy_params,      # teacher policy params
                 value_params=None): # teacher value params (optional for KL tether)
        self._make_policy = make_policy
        self._params = (normalizer_params, policy_params, value_params)
        # JIT inference function: policy(obs, key) -> (action, extras)
        self._policy = jax.jit(self._make_policy(self._params, deterministic=True))

    def act(self, obs: Dict[str, jnp.ndarray], key: jax.Array) -> Tuple[jnp.ndarray, Dict]:
        action, extras = self._policy(obs, key)
        return action, extras

