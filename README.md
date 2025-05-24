# drfc_manager
> A wrapper for DeepRacer for Cloud functionalities

![Stars](https://img.shields.io/github/stars/joaocarvoli/drfc-manager)
![GitLab Forks](https://img.shields.io/github/forks/joaocarvoli/drfc-manager)
![Contributors](https://img.shields.io/github/contributors/joaocarvoli/drfc-manager)
![Licence](https://img.shields.io/github/tag/joaocarvoli/drfc-manager)
![Issues](https://img.shields.io/github/issues/joaocarvoli/drfc-manager)
![Licence](https://img.shields.io/github/license/joaocarvoli/drfc-manager)

<img src="https://d1.awsstatic.com/deepracer/Evo%20and%20Sensor%20Launch%202020/evo-spin.fdf40252632704f3b07b0a2556b3d174732ab07e.gif" alt="EVO car" width="250">

<details open>
<summary><h1>Table of Contents</h1></summary>
  
1. [Objective](#objective)
2. [Key Advantages](#key-advantages)
3. [Usage](#usage)
4. [Idea behind](#idea-behind)

</details>
   
## Objective

The main purpose of this library is to provide an easy way to manage your workflow within the DeepRacer for Cloud (DRfC) environment. 

This library allows users to **optimize the training of their Reinforcement Learning (RL) models** by managing the entire process primarily within a Jupyter Notebook. This includes configuring model parameters, creating training pipelines, and utilizing machine learning algorithms to improve training outcomes.

## Key Advantages

- Easily set model configuration to training (hyperparameters, model metadata, and reward function)
- Create many model configurations for various training
- Stack multiple training configurations to be executed in a logical sequence

## Future ideas :bulb:
- Set a stop criteria for model training (eg. convergence, loss behavior, iteration count, or another)
- Integrate with some hyperparameter tuning techniques from machine learning libs (e.g., Sckit Learning) 

## Usage

### Define configuration model data

```python
# Default values set from official documentation
model_name = 'rl-deepracer-sagemaker'
hyperparameters = Hyperparameters() 
model_metadata = ModelMetadata()
```

### Define the reward function

```python
def reward_function():
  reward = ...
  return float(reward)
```

### Run pipeline

```python
first_train = train_pipeline(model_name, hyperparameters, model_metadata, bytes_io_reward_function)
first_train()
```

## Idea behind

This lib is being developed directly using the same ideas and implementation of 
the repo https://github.com/aws-deepracer-community/deepracer-for-cloud according to its description: _"A quick and easy way to get up and running with a DeepRacer training environment using a cloud virtual machine or a local computer"_.
