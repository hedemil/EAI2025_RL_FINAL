from dataclasses import dataclass
import jax
import jax.numpy as jnp
from flax import linen
from brax.training import networks, distribution
from brax.training.acme import running_statistics

@dataclass
class StudentPolicy:
    policy: networks.FeedForwardNetwork
    parametric: distribution.ParametricDistribution

def make_student_policy(
    obs_size,                # types.ObservationSize for a dict with key 'state'
    action_size: int,
    preprocess_fn=running_statistics.normalize,
    hidden=(128, 128),
    activation=linen.swish,
    obs_key='state',
):
    # Use the same tanh-Normal head as PPO
    parametric = distribution.NormalTanhDistribution(event_size=action_size)
    policy = networks.make_policy_network(
        param_size=parametric.param_size,
        obs_size=obs_size,
        preprocess_observations_fn=preprocess_fn,
        hidden_layer_sizes=hidden,
        activation=activation,
        obs_key=obs_key,
        distribution_type='tanh_normal',
        noise_std_type='scalar',  # matches PPO default
        init_noise_std=1.0,
        state_dependent_std=False,
    )
    return StudentPolicy(policy=policy, parametric=parametric)

# Normalizer state for a dict observation with key 'state'
def init_student_normalizer(student_obs_dim: int):
    obs_shape = {'state': jax.ShapeDtypeStruct((student_obs_dim,), jnp.float32)}
    return running_statistics.init_state(obs_shape)

