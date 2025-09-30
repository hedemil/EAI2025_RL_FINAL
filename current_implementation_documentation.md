# Current Implementation Documentation

## Overview

This document provides a comprehensive analysis of the current Go1 quadruped locomotion implementation using Brax physics simulation and PPO (Proximal Policy Optimization) reinforcement learning. The implementation is designed for training robust locomotion policies on the Go1 robot with dynamic wall obstacles and terrain variations.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Environment   │    │  Training Loop   │    │   Evaluation    │
│   (custom_env)  │◄──►│(custom_ppo_train)│◄──►│    (utils)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  MuJoCo XML     │    │   PPO Networks   │    │  Video Render   │
│  Configuration  │    │   & Loss Func    │    │  & Metrics      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## File Structure and Components

### Core Files

1. **`custom_env.py`** - Main environment implementation
2. **`custom_ppo_train.py`** - PPO training loop
3. **`utils.py`** - Visualization and evaluation utilities
4. **`train.ipynb`** - Training orchestration notebook
5. **`custom_env.xml`** - MuJoCo model definition

## Environment Implementation (`custom_env.py`)

### Class Hierarchy
```python
Joystick(go1_base.Go1Env) # Main environment class
  ├─ mjx_env.Env         # Base MuJoCo environment
  └─ go1_base.Go1Env     # Go1-specific implementations
```

### Configuration System

The environment uses a hierarchical configuration system via `ml_collections.ConfigDict`:

```python
default_config() -> config_dict.ConfigDict:
  ├─ ctrl_dt: 0.02              # Control timestep
  ├─ sim_dt: 0.004              # Simulation timestep
  ├─ episode_length: 1000       # Steps per episode
  ├─ Kp: 35.0, Kd: 0.5         # PD controller gains
  ├─ noise_config: {...}        # Sensor noise configuration
  ├─ reward_config: {...}       # Reward function weights
  ├─ pert_config: {...}         # External perturbation settings
  └─ command_config: {...}      # Command generation parameters
```

#### Noise Configuration
- **Purpose**: Adds realistic sensor noise for sim-to-real transfer
- **Components**: Joint positions, velocities, gyro, gravity, linear velocity
- **Implementation**: Uniform noise with configurable scales

#### Reward Configuration
Comprehensive reward structure with 13+ components:

**Tracking Rewards:**
- `tracking_lin_vel`: Exponential reward for linear velocity tracking
- `tracking_ang_vel`: Exponential reward for angular velocity tracking

**Stability Rewards:**
- `orientation`: Penalty for deviation from upright orientation
- `lin_vel_z`: Penalty for vertical motion
- `ang_vel_xy`: Penalty for roll/pitch angular velocities
- `torso_height`: Penalty for deviation from target height

**Locomotion Quality:**
- `pose`: Reward for maintaining default joint configuration
- `feet_air_time`: Reward for proper swing phase timing
- `feet_clearance`: Penalty for low foot clearance during swing
- `feet_slip`: Penalty for foot slipping during stance

**Energy and Smoothness:**
- `torques`: Penalty for high actuator forces
- `action_rate`: Penalty for rapid action changes
- `energy`: Penalty for mechanical power consumption

**Safety:**
- `knee_contact`: Penalty for knee collisions with ground
- `dof_pos_limits`: Penalty for approaching joint limits
- `termination`: Large penalty for episode termination

### Observation Space

The environment provides dual observation modes:

#### Standard Observations (`state` - 81 dimensions)
```python
state = jp.hstack([
    info["last_act"],                    # Previous actions (12)
    info["command"],                     # Current command (3)
    noisy_gyro,                         # Angular velocity with noise (3)
    noisy_accelerometer,                # Linear acceleration with noise (3)
    noisy_gravity,                      # Gravity vector with noise (3)
    noisy_linvel,                       # Linear velocity with noise (3)
    noisy_angvel,                       # Angular velocity with noise (3)
    noisy_joint_angles - default_pose,  # Joint angle deviations (12)
    noisy_joint_vel,                    # Joint velocities with noise (12)
    data.actuator_force,                # Motor torques (12)
    info["last_contact"],               # Previous foot contacts (4)
    feet_vel,                           # Foot velocities (12)
    info["feet_air_time"],              # Air time per foot (4)
    data.xfrc_applied[torso_body_id],   # Applied external forces (3)
    pert_active,                        # Perturbation indicator (1)
    min_height                          # Minimum terrain height (4)
])
```

#### Privileged Observations (`privileged_state` - 85 dimensions)
Same as standard observations plus additional privileged information:
- Extended terrain height information
- Ground truth sensor readings (no noise)
- Additional environmental state

### Height Mapping System

The environment implements sophisticated terrain height sensing:

```python
def height_map(self, data: mjx.Data):
    # Creates 3x3 grid around each foot
    # Ray casting to determine terrain heights
    # Returns distances and minimum heights per foot
```

**Technical Details:**
- **Grid Size**: 3×3 rays per foot (36 total rays)
- **Coverage**: ±0.1m around each foot
- **Method**: MuJoCo ray casting with robot body exclusion
- **Output**: Distance array and minimum height per foot

### Dynamic Wall System

The environment supports dynamic wall obstacles:

```python
def sample_wall_heights(self, rng, range_min=0.0, range_max=0.1):
    # Samples random wall heights per episode

def set_wall_mocap_positions(self, data, wall_heights):
    # Updates wall positions using mocap system
```

**Features:**
- **Dynamic Height**: Walls adjust height each episode
- **Curriculum Learning**: Heights can increase with training progress
- **Collision Detection**: Separate collision handling for walls vs terrain

### Command Generation

The environment uses a sophisticated command sampling system:

```python
def sample_command(self, rng, current_cmd):
    # Amplitude: Uniform distribution from command_config.a
    # Direction: Random walk for smooth transitions
    # Zero probability: command_config.b controls command diversity
```

**Command Types:**
- **Linear Velocity**: [vx, vy] in robot frame (m/s)
- **Angular Velocity**: [ωz] yaw rate (rad/s)
- **Ranges**: Configurable via `command_config.a`

### Perturbation System

External disturbances for robustness training:

```python
pert_config = config_dict.create(
    enable=False,                    # Toggle perturbations
    velocity_kick=[0.0, 3.0],       # Force magnitude range
    kick_durations=[0.05, 0.2],     # Duration range (seconds)
    kick_wait_times=[1.0, 3.0],     # Inter-perturbation intervals
)
```

## Training Implementation (`custom_ppo_train.py`)

### PPO Algorithm Modifications

The implementation is a single-device refactoring of Brax's distributed PPO:

**Key Changes:**
- Removed multi-GPU/multi-process abstractions
- Retained all core PPO functionality
- Added custom loss function injection
- Enhanced environment wrapper support

### Training Loop Structure

```python
def train(environment, num_timesteps, **kwargs):
    # 1. Environment setup with optional wrappers
    # 2. Network initialization using factory pattern
    # 3. Optimizer configuration with optional gradient clipping
    # 4. Loss function composition
    # 5. Multi-epoch training with evaluation
    # 6. Checkpoint saving and progress tracking
```

### Custom Loss Function Integration

The training loop accepts custom loss functions via `compute_custom_ppo_loss_fn`:

```python
def compute_custom_ppo_loss(
    params: PPONetworkParams,
    normalizer_params: Any,
    data: types.Transition,
    rng: jnp.ndarray,
    ppo_network: ppo_networks.PPONetworks,
    **kwargs
) -> Tuple[jnp.ndarray, types.Metrics]:
```

**Loss Components:**
1. **Policy Loss**: Clipped surrogate objective
2. **Value Loss**: Mean squared error with baseline
3. **Entropy Loss**: Exploration bonus
4. **Combined Loss**: Weighted sum of all components

### Network Architecture

The implementation uses Brax's standard PPO networks:

**Policy Network:**
- Input: Observation vector
- Hidden: Multiple fully connected layers
- Output: Action distribution parameters
- Activation: Typically tanh or ReLU

**Value Network:**
- Input: Observation vector
- Hidden: Shared or separate layers from policy
- Output: Scalar value estimate
- Loss: MSE against GAE targets

### Training Configuration

```python
ppo_training_params = {
    "num_evals": 2,                      # Evaluation frequency
    "num_timesteps": 10_000_000,         # Total training steps
    "batch_size": 32,                    # Minibatch size
    "unroll_length": 10,                 # Trajectory segment length
    "num_minibatches": 16,               # SGD minibatches per update
    "num_updates_per_batch": 2,          # PPO epochs per batch
    "learning_rate": 1e-4,               # Adam learning rate
    "entropy_cost": 1e-4,                # Entropy bonus weight
    "discounting": 0.9,                  # Discount factor
    "clipping_epsilon": 0.3,             # PPO clip parameter
    "gae_lambda": 0.95,                  # GAE λ parameter
}
```

## Loss Function Implementation

### PPO Loss Components

The custom loss function implements standard PPO with the following components:

#### 1. Policy Loss (Clipped Surrogate Objective)
```python
# Probability ratio between new and old policy
rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)

# Clipped surrogate objective
surrogate_loss1 = rho_s * advantages
surrogate_loss2 = jnp.clip(rho_s, 1 - clipping_epsilon, 1 + clipping_epsilon) * advantages
policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))
```

#### 2. Value Function Loss
```python
# MSE between value predictions and GAE targets
v_error = (vs - baseline) ** 2
v_loss = 0.25 * jnp.mean(v_error)  # 0.5 coefficient scaled down
```

#### 3. Entropy Loss
```python
# Entropy bonus for exploration
entropy = parametric_action_distribution.entropy(policy_logits, rng)
entropy_loss = -entropy_cost * jnp.mean(entropy)
```

### Generalized Advantage Estimation (GAE)

The implementation uses GAE for advantage estimation:

```python
vs, advantages = compute_gae(
    truncation=truncation,
    termination=termination,
    rewards=rewards,
    values=baseline,
    bootstrap_value=bootstrap_value,
    lambda_=gae_lambda,
    discount=discounting,
)
```

## Evaluation and Visualization (`utils.py`)

### Video Rendering System

The utility module provides comprehensive visualization capabilities:

```python
def render_video_during_training(current_policy, step_num, jit_step, jit_reset, env_cfg, eval_env):
    # 1. Policy rollout with fixed command
    # 2. Reward component tracking
    # 3. Video frame generation with overlays
    # 4. Reward plot animation
    # 5. Combined video output
```

**Features:**
- **Real-time Rendering**: During training progress
- **Reward Visualization**: Animated plots of all reward components
- **Command Visualization**: Joystick command overlays
- **Multi-group Organization**: Rewards grouped by function

### Reward Analysis

The visualization system categorizes rewards into functional groups:

```python
reward_groups = {
    'tracking': ['tracking_lin_vel', 'tracking_ang_vel'],
    'base': ['orientation', 'lin_vel_z', 'ang_vel_xy', 'pose', 'stand_still', 'torso_height'],
    'feet': ['feet_air_time', 'feet_clearance', 'feet_slip'],
    'energy': ['torques', 'action_rate', 'energy', 'termination', 'dof_pos_limits']
}
```

### Policy Evaluation

```python
def evaluate_policy(eval_env, policy_fn, jit_step, jit_reset, env_cfg, eval_env_for_video, velocity_kick_range, kick_duration_range):
    # Comprehensive policy testing across scenarios
    # Performance metrics collection
    # Robustness evaluation with perturbations
```

## Training Orchestration (`train.ipynb`)

### Jupyter Notebook Structure

The training is orchestrated through a Jupyter notebook with the following sections:

1. **Environment Setup**: Import dependencies and configure visualization
2. **Environment Creation**: Initialize custom environment with XML model
3. **PPO Implementation**: Define loss function and network architecture
4. **Training Configuration**: Set hyperparameters and callbacks
5. **Training Execution**: Run full training loop
6. **Evaluation**: Test trained policy on various scenarios

### Progress Tracking

```python
def progress(num_steps, metrics):
    # Real-time plotting of training metrics
    # Video generation during training
    # Performance tracking and visualization
```

### Policy Callback System

```python
def policy_params_callback(_, make_policy_fn, params):
    # Captures current policy for video rendering
    # Enables real-time policy visualization
    global current_policy
    current_policy = make_policy_fn(params, deterministic=True)
```

## MuJoCo Model Configuration

### XML Structure

The robot model is defined in `custom_env.xml` with the following components:

**Robot Structure:**
- Go1 quadruped with 12 actuated joints (3 per leg)
- Realistic mass and inertia properties
- Contact parameters tuned for stable simulation

**Environment Elements:**
- Plane terrain with adjustable properties
- Dynamic wall obstacles with mocap control
- Height field terrain (optional)
- Lighting and camera configuration

**Sensors:**
- Joint position and velocity sensors
- IMU (gyroscope and accelerometer)
- Foot contact sensors
- Force/torque sensors at joints

## Key Implementation Patterns

### 1. JAX-First Design
- All computations vectorized using JAX
- JIT compilation for performance
- Functional programming patterns
- Immutable data structures

### 2. Configuration-Driven Architecture
- Hierarchical configuration using ml_collections
- Easy hyperparameter modification
- Environment behavior control via config

### 3. Modular Reward Design
- Individual reward functions for each component
- Configurable reward weights
- Easy addition of new reward terms

### 4. Curriculum Learning Support
- Progressive difficulty via wall heights
- Perturbation scheduling
- Command complexity ramping

### 5. Sim-to-Real Preparation
- Realistic sensor noise modeling
- Domain randomization support
- Robust policy training

## Performance Characteristics

### Computational Requirements
- **Training Time**: ~10M timesteps for convergence
- **Memory Usage**: Moderate (single-device implementation)
- **GPU Utilization**: Efficient JAX compilation

### Simulation Fidelity
- **Physics Timestep**: 4ms simulation, 20ms control
- **Contact Modeling**: Stable ground and wall contacts
- **Sensor Realism**: Configurable noise models

### Policy Performance
- **Tracking Accuracy**: High precision command following
- **Stability**: Robust to perturbations and terrain variations
- **Generalization**: Good performance across command ranges

## Integration Points for Teacher-Student Extension

Based on this analysis, the current implementation provides excellent foundation for teacher-student learning:

### Strengths for Extension
1. **Dual Observation System**: Already supports privileged observations
2. **Modular Training Loop**: Easy to extend with additional loss terms
3. **Network Factory Pattern**: Simple to create teacher/student architectures
4. **Flexible Reward System**: Can add distillation-specific rewards
5. **JAX Infrastructure**: Efficient for complex training procedures

### Extension Opportunities
1. **Privileged Information**: Height maps, future commands, system state
2. **Network Architecture**: Separate teacher/student networks
3. **Loss Function**: Combined PPO + distillation objectives
4. **Training Pipeline**: Multi-stage teacher-student training
5. **Evaluation**: Comparative performance analysis

This comprehensive implementation provides a robust foundation for implementing teacher-student learning while maintaining the existing training infrastructure and performance characteristics.