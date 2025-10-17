# RL Final Project, overview of what we have added:
- Feet height sensors functionality to provide ground height measurements around each foot. 
- A more difficult environment that cotains stairs.
- Knee collision penalty to see if this influences the performance of the robot.
- Teacher-student policy.
- An evaluation notebook which makes it easy to try the trained models without having to train them yourself beforehand. 

## Height sensors
This works using the method height_map(). The height_map() method uses raycasting to measure distances to the ground from a 3x3 grid around each foot, providing minimum height values for all four feet. This sensor data is integrated into the observation space with noise modeling. The measurement is added to the privileged state without noise and the normal state with noise.

## Stair environment
The idea of the environment is to see how well the robot generalizes to environments not seen in training. During training the models use the normal setup following what was done in the C-level of the first lab. Then during evaluation we can both assess how well the robot behaves in the same environment with different commands, and then test in the stairs environment. This enables more thorough evaluation of the trained models.

## Knee collision
This follows what was done for monitoring feet collisions. We check for when geometry ids of the knees intersect with geometry ids of the walls and flag this as a knee collision.

## Teacher-student policy
The idea is that we use a model trained with PPO as the teacher and train a smaller network containing one neural network such as MLP or RNN to output the same action distribution as the teacher given an observation with less information. This is done using imitation learning. The teacher and student are provided information from an observation where the teacher has more and perfect data and the student has less and noisy data. Then the outputs of the networks are compared, and the student is trained to output the same action distribution as the teacher. What we hope to see is that the student can find the most important bits of information from the observation that the teacher might not recognize in its abundance of information, or at least that the student can achieve good performance with a smaller network.

## Evaluation notebook
- List all available parameter files (.npy format)
- Loads the network configuration matching the training setup
- Runs evaluation episodes with different velocity commands and pertubations

<!-- RESULTS-START -->
## Results

This section showcases visual results. Add your GIFs to the folders under `results/` and embed them below using relative paths. Recommended width is 280–320 px for GitHub readability.

Naming guidelines (recommended for consistency)
- Use subfolders: `results/baseline/`, `results/height_map/`, `results/knee_collision/`, `results/student_teacher/`.
- Order GIFs with leading indices: `00_*.gif`, `01_*.gif`, …
- Keep 2–4 concise GIFs per section.

### Baseline
Place GIFs in `results/baseline/` and embed here.

<!-- Example: replace with your GIFs -->
<p align="center">
  <img src="results/student_bc_mse/teacher_student_0.gif" width="280" alt="Baseline: Forward" />
  <!-- <img src="results/baseline/01_turn.gif" width="280" alt="Baseline: Turn" /> -->
  <!-- <img src="results/baseline/02_stairs.gif" width="280" alt="Baseline: Stairs" /> -->
  <!-- <img src="results/baseline/03_recovery.gif" width="280" alt="Baseline: Recovery" /> -->
  <!-- Remove the example block above and insert your files -->
</p>

### Height Map
Place GIFs in `results/height_map/` and embed here.

<!-- Example: replace with your GIFs -->
<!-- <p align="center">
  <img src="results/height_map/00_flat.gif" width="280" alt="Height Map: Flat" />
  <img src="results/height_map/01_rough.gif" width="280" alt="Height Map: Rough" />
</p> -->

### Knee Collision
Place GIFs in `results/knee_collision/` and embed here.

<!-- Example: replace with your GIFs -->
<!-- <p align="center">
  <img src="results/knee_collision/00_low_wall.gif" width="280" alt="Knee Collision: Low Wall" />
  <img src="results/knee_collision/01_high_wall.gif" width="280" alt="Knee Collision: High Wall" />
</p> -->

### Student–Teacher
Place GIFs in `results/student_teacher/` and embed here.

<!-- Example: replace with your GIFs -->
<!-- <p align="center">
  <img src="results/student_teacher/00_teacher.gif" width="280" alt="Teacher" />
  <img src="results/student_teacher/01_student.gif" width="280" alt="Student" />
  <img src="results/student_teacher/02_side_by_side.gif" width="280" alt="Teacher vs Student" />
</p> -->
<!-- RESULTS-END -->


# RL Final Project Starter Code

## Overview
We have extended the codebase of Lab 1 to help you get started with the final project and to avoid you having to search for some functionalities in the depth of mujoco :)

## Codebase changes
The main functionalities we implemented for you are:
- Moved the environment definition (previously `joystick.py` in the library source) to this repo. You can find it under `custom_env.py` and you will make your changes to the environment (reward functions, feet height sensors, ...) there
- We have moved the task specification (where are the walls) into this repo as well, see `custom_env.xml`. We have also added a `custom_env_debug_walls.py` which adds an extra wall right under the robot to debug functionality you might implement.
- We have added **knee collisions** to the simulation, and provided you knee IDs to allow you to formulate penalties for knee collisions, if you want.
- We have added **feet height sensors** functionality to provide ground height measurements around each foot. The `height_map()` method uses raycasting to measure distances to the ground from a 3x3 grid around each foot, providing minimum height values for all four feet. This sensor data is integrated into the observation space with noise modeling.
- We have added a *environment randomization/curriculum learning* capabilities to the environment setup. More specifically, this allows you to programatically update the wall height to control the environment's difficulty during training. See the relevant code pieces in `custom_env.py`. We achieve this through the defining the walls as *mocap bodies* which are bodies that can be moved around, but do not have physics (i.e. they are unmovable by collisions or gravity).
   - Note that wall heights are currently set randomly upon environment reset, you will likely want to replace this by something based on training progres
- We have added a script, `visualize_custom_env.py`, that simulates the environment over a few resets to make sure that walls are updated are expected. This also makes sure that wall updates affect the physics correctly. Lastly, this script reads the `torso_height` measurements and overlays it in the video. This should give you an idea on how to implement the feet height scanners. This script also shows you how to load our custom environment, rather than through mujoco's `registry`.

### Evaluation Notebook (`evaluation/eval_params.ipynb`)
  - Lists all available parameter files (`.npy` format)
  - Loads the network configuration matching the training setup
  - Runs evaluation episodes with different velocity commands and petrubations




## Getting Started
We will provide instructions to use our GPU cluster in the lab (or use your own machine). This mostly follows the lab instructions, but you will need a different version of `mujoco_playground`, see below.
### Clone the repository

First, clone the starter code repo:

```git clone https://github.com/finnBsch/eai2025_rl_final.git```

Then navigate to the cloned directory:

```cd eai2025_rl_final```

### Create a Python Virtual Environment
(pretty much same as lab)

Run: 
```python3 -m venv <your_venv_name>```

For example:

```python3 -m venv venv_rl```

Then run:

```source <path_to_your_venv>/venv_rl/bin/activate```


### Install general dependencies

With your venv sourced please run the following commands in the terminal:
```
python3 -m pip install matplotlib opencv-python mediapy jax[cuda12]
```

### Install mujoco related dependencies

Furthermore install these dependencies (note: the order matters here, so please stick to it, if something seems to go wrong, just restart from the first one and run them all again):
```
python3 -m pip install git+https://github.com/finnBsch/mujoco_playground.git@final_proj
python3 -m pip uninstall mujoco-mjx
python3 -m pip install git+https://github.com/finnBsch/mujoco.git@lab#subdirectory=mjx
```
**Note that the `mujoco_playground` version here is different from the lab!**
## Related Resources

- PPO paper (OpenAI): https://arxiv.org/pdf/1707.06347
- Perceptive Quadruped RL (ETH Zurich): https://arxiv.org/pdf/2201.08117
- Non-perceptive Quadruped RL (ETH Zurich): https://arxiv.org/pdf/2010.11251
