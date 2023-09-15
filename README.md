# Exploring the Impact of Human Feedback on Reinforcement Learning for Game Playing AI Agents

## Overviews

This repository contains the implementation of my Final Project for my Bachelor's in Computer Science at the University of London. The project's primary objective is to assess the influence of human feedback on reinforcement learning, with a specific focus on training efficiency and overall performance quality. To achieve this, I will conduct a comparative analysis of two AI systems: one based on conventional reinforcement learning and another that integrates a reward function learned from human gameplay. Subsequently, both systems will undergo further refinement using traditional reinforcement learning techniques. The evaluation will take place in the context of the popular game Doom, using the ViZDoom tool, and will rely solely on visual input from the game environment.

## Usage Guide

### Setting up the Project Locally

The dependenices of the project have been setup as an Anaconda environment to make it easier for recreating the environement. At the moment, the project was only tested on Windows x64, but it will most likely work on Linux x86-64bit as well. It is recommended to anyone trying it locally to run it on an Windows x64 machine. 

To begin, one has to first install the following applications on the machine:

* Anaconda
* Python 3.9.7

Once installed the environment can be setup using either one of the provided environment files, namely `x86_64-cpu-environment.yml` and `x86_64-cuda-environment.yml` depending upon the availabilty of NVIDIA Graphics Card. It must be noted that in order to use the GPU environment, the installed NVIDIA graphics card should support CUDA. To install the dependencies one of the following command should be executed from the root folder of the project folder in the terminal:

```shell
# To create CPU environment
conda env create --file x86_64-cpu-environment.yml

# To create CUDA environment
conda env create --file x86_64-cuda-environment.yml
```

Once the environment is setup, the environment can be activated by running the following command in the terminal. 

```shell
conda activate doom-rlhf
```

### Testing Trained Models

#### Prerequsites

The `play_doom_with_trained_agent.py` script requires the Anaconda Environment to be setup and activated. More information regarding the setup process can be found in the [local setup](#appendix.usage-guide.local-setup) section above.

#### How to Run

The `play_doom_with_trained_agent.py` script has been developed as a command line script and can simply be executed using the following command

```shell
python play_doom_with_trained_agent.py --env-cfg envs/vizdoom/scenarios/basic.cfg --agent-model final_models/baseline_rl_model --episodes 50 --enable-gpu
```

Among the different arguments used in the above command, `--env-cfg` and `--agent-model` are the required arguments while the number of episodes is set to `50` by default, and the GPU is disabled by default. These parameters will be listed along with their description using the `--help` argument when executing the script and for reference they have also been listed in the table below.

| Argument           | Description                                  |
|--------------------|----------------------------------------------|
| --env-cfg          | Path to ViZDoom level config                 |
| --agent-model      | Path to trained model                        |
| --episodes         | Number of episodes to run the agent for     |
| --enable-gpu       | Flag for enabling GPU usage for the neural networks |

#### Available Models

The following models can be used with the `play_doom_with_trained_agent.py` script:

* **RL Pipeline Trained Doom PPO Agent Model (Baseline Model):** `./final_models/baseline_doom_ppo_agent_model`
* **RLHF Pipeline Trained Doom PPO Agent Model:** `./final_models/rlhf_pipeline_trained_doom_ppo_agent_model`

### Running Reinforcement Learning Pipeline

#### Prerequsites

The `run_doom_rl_pipeline.py` script requires the Anaconda Environment to be setup and activated. More information regarding the setup process can be found in the [local setup](#appendix.usage-guide.local-setup) section above.

#### How to Run

Being implemented as a command line interface (CLI), the `run_doom_rl_pipeline.py` script supports the customization of hyper-parameters for the training process using command line arguments. The arguments supported by the script along with their default values can be viewed by passing the `--help` argument when running the script using python. These arguments are also listed and described in the table presented below.

| Argument                 | Description                                           |
|--------------------------|-------------------------------------------------------|
| --env-cfg                | Path to ViZDoom level config                          |
| --learning-rate          | Learning rate of the RL agent                        |
| --total-timesteps        | Total timesteps the agent should run for             |
| --render-env             | Flag for rendering one of the multiple environments  |
| --enable-gpu             | Flag for enabling GPU usage for the neural networks  |
| --track-stats            | Track stats using Tensorboard                        |
| --num-envs               | Number of environments to run simultaneously        |
| --num-steps              | Number of steps per environment for a training batch |
| --anneal-lr              | Flag for annealing the learning rate of the RL agent |
| --enable-gae             | Flag for enabling GAE calculation during RL agent training |
| --gamma                  | Coefficient that determines the influence of future rewards |
| --gae-lambda             | The lambda value for GAE calculation during RL agent training |
| --num-minibatches        | Number of mini-batches to use when training RL agent |
| --training-epochs        | Number of epochs to train the RL agent with a training batch |
| --norm-adv               | Flag to normalize calculated advantages during RL agent training |
| --clip-coef              | The clipping coefficient of losses when training RL agent |
| --clip-vloss             | Flag for clipping value loss when training RL agent |
| --entropy-coef           | The coefficient for the entropy loss when training RL agent |
| --value-coef             | The coefficient for the value loss when training RL agent |
| --max-grad-norm          | Maximum normalized gradient value when training RL agent |
| --target-kl              | Maximum KL divergence limit before stopping RL agent training |

The script can be executed using python using the following command:

```shell
python run_doom_rl_pipeline.py --env-cfg envs/vizdoom/scenarios/basic.cfg
```

It must be noted that `--env-cfg` is a mandatory argument and the script requires it for functioning. The above example uses the VizDoom level config file located at `envs/vizdoom/scenarios/basic.cfg` since this is the level being used throughout the project. However, the other `'.cfg'` files located in the `envs/vizdoom/scenarios` directory will work fiine as well.

### Reinforcement Learning using Human Feedback Pipeline

#### Prerequsites

The `run_doom_rl_pipeline.py` script requires the Anaconda Environment to be setup and activated. More information regarding the setup process can be found in the [local setup](#appendix.usage-guide.local-setup) section above. Once the environment is activated a Jupyter Notebook server can be started in the root directory of the project, which can then be used to run the pipeline. 

#### How to Run

Being implemented as a Jupyter Notebook, the RLHF pipeline does not support arguments like the RL pipeline script and will require change to the params in the notebook itself for customization. However this process has been made easier by initializing the params for the pipeline using dictionaries in a separate code cell. There are three config dictionaries each corresponding to a different component of the pipeline. These configuration keys are listed in [Table @tbl:rlhf-pipeline-arguments]. Once the configuration of the pipeline is done, it can be executed by simply running all the cells of the notebook in order.

| Config Key            | Dictionary Name   | Description                                               |
|-----------------------|--------------------|-----------------------------------------------------------|
| env_cfg               | pipeline_args      | Path to ViZDoom level config                              |
| total_timesteps       | pipeline_args      | Total timesteps the agent should run for                   |
| render_env            | pipeline_args      | Flag for rendering one of the multiple environments        |
| enable_gpu            | pipeline_args      | Flag for enabling GPU usage for the neural networks        |
| track_stats           | pipeline_args      | Track stats using Tensorboard                              |
| num_envs              | pipeline_args      | Number of environments to run simultaneously              |
| env_replay_buffer_size| pipeline_args      | Number of steps per environment for a training batch       |
| enable_pre_training   | pipeline_args      | Flag for enabling the pre-training phase                   |
| num_pre_train_requests| pipeline_args      | Number of preferences to collect during pre-training       |
| num_trajectory_frames | pipeline_args      | Number of frames in trajectory video                        |
| human_feedback_interval| pipeline_args      | Number of agent training steps before collecting human feedback in the training phase |
| learning_rate         | agent_args         | Learning rate of the RL agent                              |
| anneal_lr             | agent_args         | Flag for annealing the learning rate of the RL agent      |
| enable_gae            | agent_args         | Flag for enabling GAE calculation during RL agent training |
| gamma                 | agent_args         | Coefficient that determines the influence of future rewards |
| gae_lambda            | agent_args         | The lambda value for GAE calculation during RL agent training |
| num_minibatches       | agent_args         | Number of mini-batches to use when training RL agent       |
| num_training_epochs   | agent_args         | Number of epochs to train the RL agent with a training batch |
| norm_adv              | agent_args         | Flag to normalize calculated advantages during RL agent training |
| clip_coef             | agent_args         | The clipping coefficient of losses when training RL agent  |
| clip_vloss            | agent_args         | Flag for clipping value loss when training RL agent        |
| entropy_coef          | agent_args         | The coefficient for the entropy loss when training RL agent |
| value_coef            | agent_args         | The coefficient for the value loss when training RL agent  |
| max_grad_norm         | agent_args         | Maximum normalized gradient value when training RL agent   |
| target_kl             | agent_args         | Maximum KL divergence limit before stopping RL agent training |
| learning_rate         | reward_predictor_args | Learning rate for the Reward Predictor                 |
| hidden_layer_size     | reward_predictor_args | Size of hidden layers in the reward predictor           |
|
