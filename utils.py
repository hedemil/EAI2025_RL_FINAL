import matplotlib.pyplot as plt
import jax
import mujoco
import numpy as np
import jax.numpy as jp
import functools
import cv2
import mediapy as media
from mujoco_playground._src.gait import draw_joystick_command

def render_video_during_training(current_policy, step_num, jit_step,jit_reset, env_cfg, eval_env_for_video):
    """Render a video using the current policy during training"""
    try:
        jit_inference_fn = jax.jit(current_policy)
        from mujoco_playground._src.gait import draw_joystick_command
        
        # Use a simple forward command for training videos
        command = jp.array([1.0, 0.0, 0.0])  # Move forward
        
        rng = jax.random.PRNGKey(42)  # Fixed seed for consistency
        rollout = []
        modify_scene_fns = []
        reward_history = []
        
        state = jit_reset(rng)
        state.info["command"] = command
        
        rollout_length = min(160, env_cfg.episode_length)
        
        for _ in range(rollout_length):
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)
            state.info["command"] = command
            rollout.append(state)
            
            raw_rewards = {k[7:]: v for k, v in state.metrics.items() if k.startswith("reward/")}
            scaled_rewards = {}
            for k, v in raw_rewards.items():
                scaled_rewards[k] = v
            reward_history.append(scaled_rewards)
            
            # Add visualization elements
            xyz = np.array(state.data.xpos[eval_env_for_video._torso_body_id])
            xyz += np.array([0, 0, 0.2])
            x_axis = state.data.xmat[eval_env_for_video._torso_body_id, 0]
            yaw = -np.arctan2(x_axis[1], x_axis[0])
            modify_scene_fns.append(
                functools.partial(
                    draw_joystick_command,
                    cmd=state.info["command"],
                    xyz=xyz,
                    theta=yaw,
                    scl=abs(state.info["command"][0]) / env_cfg.command_config.a[0],
                )
            )

        render_every = 2
        fps = 1.0 / eval_env_for_video.dt / render_every
        traj = rollout[::render_every]
        mod_fns = modify_scene_fns[::render_every]

        scene_option = mujoco.MjvOption()
        scene_option.geomgroup[2] = True
        scene_option.geomgroup[3] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True

        frames = eval_env_for_video.render(
            traj,
            camera="track",
            scene_option=scene_option,
            width=640,
            height=480,
            modify_scene_fns=mod_fns,
        )
        
        # Get reward component keys from first frame
        reward_keys = set()
        if reward_history:
            reward_keys = set(reward_history[0].keys())
        
        # Calculate global min/max values for consistent y-axis scaling
        def calculate_global_limits(reward_history, reward_groups, reward_keys):
            """Calculate global min/max for each reward group from all data"""
            group_limits = {}
            
            for group_name, reward_names in reward_groups.items():
                all_values = []
                for reward_name in reward_names:
                    if reward_name in reward_keys:
                        values = [reward_history[i].get(reward_name, 0.0) for i in range(len(reward_history))]
                        all_values.extend(values)
                
                if all_values:
                    min_val = min(all_values)
                    max_val = max(all_values)
                    # Add some padding (10% of range)
                    range_val = max_val - min_val
                    padding = max(0.1 * range_val, 0.1)  # At least 0.1 padding
                    group_limits[group_name] = (min_val - padding, max_val + padding)
                else:
                    group_limits[group_name] = (-1, 1)  # Default range
            
            return group_limits
        
        # Define reward groups
        reward_groups = {
            'tracking': ['tracking_lin_vel', 'tracking_ang_vel'],
            'base': ['orientation', 'lin_vel_z', 'ang_vel_xy', 'pose', 'stand_still', 'torso_height'],
            'feet': ['feet_air_time', 'feet_clearance', 'feet_slip'],
            'energy': ['torques', 'action_rate', 'energy', 'termination', 'dof_pos_limits']
        }
        
        # Calculate global limits once for all frames
        global_limits = calculate_global_limits(reward_history, reward_groups, reward_keys)
        
        # Create reward plots for each frame
        def create_reward_plot(frame_idx, reward_history, reward_keys):
            """Create animated reward plot for a specific frame"""
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))
            fig.suptitle(f'Reward Components - Step {step_num} - Frame {frame_idx}', fontsize=14)
            
            # Prepare data up to current frame
            current_step = min(frame_idx * render_every, len(reward_history) - 1)
            steps = list(range(0, current_step + 1))
            
            if current_step >= 0:
                # Plot 1: Tracking rewards
                for reward_name in reward_groups['tracking']:
                    if reward_name in reward_keys:
                        values = [reward_history[i].get(reward_name, 0.0) for i in range(current_step + 1)]
                        ax1.plot(steps, values, label=reward_name, linewidth=2)
                ax1.set_title('Tracking Rewards')
                ax1.legend(fontsize=8)
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim(global_limits['tracking'])
                
                # Plot 2: Base stability rewards
                for reward_name in reward_groups['base']:
                    if reward_name in reward_keys:
                        values = [reward_history[i].get(reward_name, 0.0) for i in range(current_step + 1)]
                        ax2.plot(steps, values, label=reward_name, linewidth=2)
                ax2.set_title('Base Stability')
                ax2.legend(fontsize=8)
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(global_limits['base'])
                
                # Plot 3: Feet rewards
                for reward_name in reward_groups['feet']:
                    if reward_name in reward_keys:
                        values = [reward_history[i].get(reward_name, 0.0) for i in range(current_step + 1)]
                        ax3.plot(steps, values, label=reward_name, linewidth=2)
                ax3.set_title('Feet Control')
                ax3.legend(fontsize=8)
                ax3.grid(True, alpha=0.3)
                ax3.set_ylim(global_limits['feet'])
                
                # Plot 4: Energy/Action rewards
                for reward_name in reward_groups['energy']:
                    if reward_name in reward_keys:
                        values = [reward_history[i].get(reward_name, 0.0) for i in range(current_step + 1)]
                        ax4.plot(steps, values, label=reward_name, linewidth=2)
                ax4.set_title('Energy & Actions')
                ax4.legend(fontsize=8)
                ax4.grid(True, alpha=0.3)
                ax4.set_ylim(global_limits['energy'])
            
            plt.tight_layout()
            
            # Convert plot to image
            fig.canvas.draw()
            # Use buffer_rgba() instead of deprecated tostring_rgb()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            plot_image = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            # Convert RGBA to RGB
            plot_image = plot_image[:, :, :3]
            plt.close(fig)
            
            return plot_image
        
        # Create combined frames (video + reward plot side by side)
        combined_frames = []
        for frame_idx, frame in enumerate(frames):
            # Create reward plot for this frame
            reward_plot = create_reward_plot(frame_idx, reward_history, reward_keys)
            
            # Resize frames to match heights
            video_height = frame.shape[0]
            plot_height, plot_width = reward_plot.shape[:2]
            
            # Scale plot to match video height
            if plot_height != video_height:
                scale_factor = video_height / plot_height
                new_plot_width = int(plot_width * scale_factor)
                reward_plot = cv2.resize(reward_plot, (new_plot_width, video_height))
            
            # Combine frames horizontally
            combined_frame = np.hstack([frame, reward_plot])
            combined_frames.append(combined_frame)
        
        media.show_video(combined_frames, fps=fps)
        
    except Exception as e:
        print(f"Failed to render video at step {step_num}: {e}")


def evaluate_policy(
    env,
    jit_inference_fn,
    jit_step,
    jit_reset,
    env_cfg,
    eval_env,
    velocity_kick_range: tuple = (-1.0, 1.0),
    kick_duration_range: tuple = (0.2, 1.0),
):
    """Evaluate the policy over multiple episodes and return average reward."""
    x_vels = [0.0, 0.5, 1.0, 0.0, 0.0, 0.5]
    y_vels = [0.0, 0.0, 0.0, 0.5, 1.0, 0.5]
    yaw_vels = [1.0, 0.0, 0.0, 0.0, 0.0, 0.1]

    for run_id, (x_vel, y_vel, yaw_vel) in enumerate(zip(x_vels, y_vels, yaw_vels)):
        def sample_pert(rng):
            rng, key1, key2 = jax.random.split(rng, 3)
            pert_mag = jax.random.uniform(
                key1, minval=velocity_kick_range[0], maxval=velocity_kick_range[1]
            )
            duration_seconds = jax.random.uniform(
                key2, minval=kick_duration_range[0], maxval=kick_duration_range[1]
            )
            duration_steps = jp.round(duration_seconds / eval_env.dt).astype(jp.int32)
            state.info["pert_mag"] = pert_mag
            state.info["pert_duration"] = duration_steps
            state.info["pert_duration_seconds"] = duration_seconds
            return rng


        rng = jax.random.PRNGKey(0)
        rollout = []
        modify_scene_fns = []

        swing_peak = []
        rewards = []
        linvel = []
        angvel = []
        track = []
        foot_vel = []
        rews = []
        contact = []
        command = jp.array([x_vel, y_vel, yaw_vel])

        state = jit_reset(rng)
        if state.info["steps_since_last_pert"] < state.info["steps_until_next_pert"]:
            rng = sample_pert(rng)
        state.info["command"] = command
        for _ in range(env_cfg.episode_length):
            if state.info["steps_since_last_pert"] < state.info["steps_until_next_pert"]:
                rng = sample_pert(rng)
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)
            state.info["command"] = command
            rews.append(
                {k: v for k, v in state.metrics.items() if k.startswith("reward/")}
            )
            rollout.append(state)
            swing_peak.append(state.info["swing_peak"])
            rewards.append(
                {k[7:]: v for k, v in state.metrics.items() if k.startswith("reward/")}
            )
            linvel.append(env.get_global_linvel(state.data))
            angvel.append(env.get_gyro(state.data))
            track.append(
                env._reward_tracking_lin_vel(
                    state.info["command"], env.get_local_linvel(state.data)
                )
            )

            feet_vel = state.data.sensordata[env._foot_linvel_sensor_adr]
            vel_xy = feet_vel[..., :2]
            vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
            foot_vel.append(vel_norm)

            contact.append(state.info["last_contact"])

            xyz = np.array(state.data.xpos[env._torso_body_id])
            xyz += np.array([0, 0, 0.2])
            x_axis = state.data.xmat[env._torso_body_id, 0]
            yaw = -np.arctan2(x_axis[1], x_axis[0])
            modify_scene_fns.append(
                functools.partial(
                    draw_joystick_command,
                    cmd=state.info["command"],
                    xyz=xyz,
                    theta=yaw,
                    scl=abs(state.info["command"][0])
                    / env_cfg.command_config.a[0],
                )
            )


        render_every = 2
        fps = 1.0 / eval_env.dt / render_every
        traj = rollout[::render_every]
        mod_fns = modify_scene_fns[::render_every]

        scene_option = mujoco.MjvOption()
        scene_option.geomgroup[2] = True
        scene_option.geomgroup[3] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True

        frames = eval_env.render(
            traj,
            camera="track",
            scene_option=scene_option,
            width=640,
            height=480,
            modify_scene_fns=mod_fns,
        )
        media.show_video(frames, fps=fps)