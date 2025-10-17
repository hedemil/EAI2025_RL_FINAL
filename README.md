# RL Final Project
This project was carried out by

## Overview
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

# How to run the code
This will go over how to get started and then run the code

## Getting Started
We will provide instructions to use our GPU cluster in the lab (or use your own machine). This mostly follows the lab instructions, but you will need a different version of `mujoco_playground`, see below.
### Clone the repository

First, clone the starter code repo:

```git clone https://github.com/hedemil/EAI2025_RL_FINAL.git```

Then navigate to the cloned directory:

```cd EAI2025_RL_FINAL```

### Create a Python Virtual Environment

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


## Running the code
This will go over how to run the code

### Training PPO network
- In [training notebook](train.ipynb) set the environment xml file. Default is custom_env.xml. Another option we added is stairs_env.xml.
- In [custom python env](environments/custom_env.py) two things can be changed here. First, choose if height information should be in privileged and/or normal state or comment it out. Default is that it is included. Second, if knee collision should be used set the reward config to have a negative value (we have used -0.3). Default is that it is on. To turn it off set it to 0.0.
- In [training notebook](train.ipynb) the network that is trained will be saved in [parameters folder](parameters) with the file name that is set in the notebook.
- Now the training can be ran. By default it will be evaluated in the custom environment after finishing training. Training is set to 150M environment steps and 10 evaluations.

### Evaluation notebook for PPO network
In the [evaluation notebook](evaluation/eval_params.ipynb) it is possible to test the networks that we have trained. The notebook will:
- List all available parameter files (.npy format)
- Loads the network configuration matching the training setup
- Runs evaluation episodes with different velocity commands and pertubations

### Teacher-student policy
It is possible to run the training with default configurations, otherwise these can be changed in the [teacher student notebook](training/teacher_student_MLP.ipynb):
- xml_path: path to the environment XML. Defaults to ../environments/custom_env.xml. You can switch to ../environments/stairs_env.xml.
- student_observation_key: choose the student input features. Options: "student_state", "state", "privileged_state" (for analysis). Default: "state".
- data_collection: select data source. Options: "bc" (behavior cloning, teacher rollouts) or "dagger" (student rollouts labeled by teacher). Default: "dagger".
- loss_function_name: imitation loss. Options: "mse" (mean‑squared error on means) or "kl" (KL between Normal(mu, sigma)). Default: "mse". MSE: minimize squared error between student means and teacher means on the same observation batch. KL: closed‑form KL for diagonal Normals, per action dimension, pre‑tanh: $KL = \log(\frac{\sigma_t}{\sigma_s}) + \frac{\sigma_s^2 + (\mu_s−\mu_t)^2}{2 \sigma_t^2} − 0.5$.
- batch_size: training batch size. Default: 64.
- learning_rate: optimizer LR (AdamW). Default: 1e-4.
- experiment_name: auto‑built as student_{data_collection}_{loss_function_name}_{student_observation_key}.
- experiment_path: auto‑built as ../results/{experiment_name}; created if missing.
- Teacher params: loaded from ../parameters/params_with_height_and_knee.npy. You can change to any file in parameters/ (e.g., params_baseline.npy, params_with_height.npy, teacher_params.npy).

Now run the code
- Run [teacher student notebook](training/teacher_student_MLP.ipynb)
- Results (GIFs and params) are saved in [results folder](results)







