# Formatting rules for logs:
* **Dates**
    * yy/mm/dd
* **File**
    * Relative path
* **Function/Code Block**
    * Part of code, if relevant
*   **Description**
    * Short description of what has been done
# Example:
# Date - File - Function/Code Block
- Description

# 25/09/18 - PROJECT_LOG.md - None
- Initialized Project Log

# 25/09/23 - PROJECT_LOG.md - None
- Set up environment in cluster to be able to train, got help from TAs to do it, was a version mismatch in mujoco and mujoco-mjx
- Loke and Malte implemented height measurements from each feet. This is based on ray tracing down from the feet. Each foot has a grid of start positions to make it more robust.
- Emil and Andrea set up training in the environment, reusing code from lab 1 so that it is possible to see reward graph and evaluation during training.

# 25/09/24 - PROJECT_LOG.md - None
- Emil and Andrea find a solution to the problem of floating obstacles.
- Malte and Loke tested the height readings in a simple environment and concluded that it seemed to work.
- Started training with height sensors to see if it makes a difference.

# 25/09/25 - train.ipynb - visualization of progress function
- changed x limits to plt.xlim([0, ppo_training_params["num_timesteps"] * 1.25]) so it used the configured num_timesteps. This is used to display the reward graph.
- changed number of evals to 25 and steps to 100M

# 25/09/25 - custom_env.py - reward config
- changed config to be the same as Malte's C level task from lab 1 