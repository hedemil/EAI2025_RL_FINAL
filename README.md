# RL Final Project Starter Code

## Overview
We have extended the codebase of Lab 1 to help you get started with the final project and to avoid you having to search for some functionalities in the depth of mujoco :)

## Codebase changes
The main functionalities we implemented for you are:
- Moved the environment definition (previously `joystick.py` in the library source) to this repo. You can find it under `custom_env.py` and you will make your changes to the environment (reward functions, feet height sensors, ...) there
- We have moved the task specification (where are the walls) into this repo as well, see `custom_env.xml`. We have also added a `custom_env_debug_walls.py` which adds an extra wall right under the robot to debug functionality you might implement.
- We have added **knee collisions** to the simulation, and provided you knee IDs to allow you to formulate penalties for knee collisions, if you want.
- We have added a *environment randomization/curriculum learning* capabilities to the environment setup. More specifically, this allows you to programatically update the wall height to control the environment's difficulty during training. See the relevant code pieces in `custom_env.py`. We achieve this through the defining the walls as *mocap bodies* which are bodies that can be moved around, but do not have physics (i.e. they are unmovable by collisions or gravity).
   - Note that wall heights are currently set randomly upon environment reset, you will likely want to replace this by something based on training progres
- We have added a script, `visualize_custom_env.py`, that simulates the environment over a few resets to make sure that walls are updated are expected. This also makes sure that wall updates affect the physics correctly. Lastly, this script reads the `torso_height` measurements and overlays it in the video. This should give you an idea on how to implement the feet height scanners. This script also shows you how to load our custom environment, rather than through mujoco's `registry`.

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
