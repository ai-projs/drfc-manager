# drfc_manager
> A Pythonic workflow manager and wrapper for DeepRacer for Cloud (DRfC)

![Stars](https://img.shields.io/github/stars/joaocarvoli/drfc-manager)
![GitLab Forks](https://img.shields.io/github/forks/joaocarvoli/drfc-manager)
![Contributors](https://img.shields.io/github/contributors/joaocarvoli/drfc-manager)
![Licence](https://img.shields.io/github/tag/joaocarvoli/drfc-manager)
![Issues](https://img.shields.io/github/issues/joaocarvoli/drfc-manager)
![Licence](https://img.shields.io/github/license/joaocarvoli/drfc-manager)
![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/ai-projs/drfc-manager?utm_source=oss&utm_medium=github&utm_campaign=ai-projs%2Fdrfc-manager&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)

<img src="https://d1.awsstatic.com/deepracer/Evo%20and%20Sensor%20Launch%202020/evo-spin.fdf40252632704f3b07b0a2556b3d174732ab07e.gif" alt="EVO car" width="250">

<details open>
<summary><h1>Table of Contents</h1></summary>
  
1. [Objective](#objective)
2. [Key Features](#key-features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)
7. [Idea behind](#idea-behind)

</details>
   
## Objective

The main purpose of this library is to provide a **Pythonic, Jupyter-friendly interface** to manage your workflow within the DeepRacer for Cloud (DRfC) environment. 

This library allows users to **optimize the training, evaluation, and management of Reinforcement Learning (RL) models** by orchestrating the entire process from Python scripts or Jupyter Notebooks. It supports local, MinIO, and AWS S3 storage, and is designed for multi-user environments (e.g., JupyterHub).

## Key Features

- **Easy model configuration** for training (hyperparameters, model metadata, reward function)
- **Pipeline management** for training, evaluation, cloning, stopping, and metrics
- **Multi-user support**: user-specific temp/log directories, safe for JupyterHub
- **Local, MinIO, and AWS S3** storage support
- **Automatic Docker Compose orchestration** for all flows
- **Jupyter Notebook integration**: run, monitor, and stop jobs from notebooks
- **Advanced logging**: per-user, per-run logs for debugging and reproducibility
- **Extensible**: add new pipeline steps or customize existing ones
- **Integrated viewer pipeline**: Launches a real-time Streamlit-based viewer and video stream proxy for model evaluation and monitoring.

## Installation

```bash
pip install drfc_manager
# or clone and install locally
# git clone https://github.com/joaocarvoli/drfc-manager.git
# cd drfc-manager && pip install .
```

## Usage

### 1. Define configuration model data

```python
from drfc_manager.types.hyperparameters import HyperParameters
from drfc_manager.types.model_metadata import ModelMetadata

model_name = 'rl-deepracer-sagemaker'
hyperparameters = HyperParameters() 
model_metadata = ModelMetadata()
```

### 2. Define the reward function

```python
def reward_function(params):
    # Your custom reward logic here
    return float(...)
```

### 3. Run a training pipeline

```python
from drfc_manager.pipelines import train_pipeline

train_pipeline(
    model_name=model_name,
    hyperparameters=hyperparameters,
    model_metadata=model_metadata,
    reward_function=reward_function,
    overwrite=True,
    quiet=False
)
```

### 4. Evaluate a model

```python
from drfc_manager.pipelines import evaluate_pipeline

result = evaluate_pipeline(
    model_name=model_name,
    run_id=0,
    quiet=True,
    clone=False,
    save_mp4=True
)
```

### 5. Clone a model

```python
from drfc_manager.pipelines import clone_pipeline

clone_pipeline(
    model_name=model_name,
    new_model_name='my-cloned-model',
    quiet=True
)
```

### 6. Stop a running pipeline

```python
from drfc_manager.pipelines import stop_training_pipeline, stop_evaluation_pipeline

stop_training_pipeline(run_id=0)
stop_evaluation_pipeline(run_id=0)
```

### 7. Start/Stop metrics (Grafana/Prometheus)

```python
from drfc_manager.pipelines import start_metrics_pipeline, stop_metrics_pipeline

start_metrics_pipeline(run_id=0)
stop_metrics_pipeline(run_id=0)
```

### 8. Start/Stop the viewer

```python
from drfc_manager.pipelines import start_viewer_pipeline, stop_viewer_pipeline

# Start the viewer (for a given run_id)
viewer_result = start_viewer_pipeline(run_id=0, quiet=True)

# Stop the viewer
stop_viewer_pipeline(quiet=True)
```

## Advanced Usage

- **Multi-user JupyterHub:** All logs and temp files are stored in user-specific directories under `/tmp/<username>/`.
- **Environment Variables:** You can override any DRfC or DeepRacer environment variable by setting it in your `.env` or before running a pipeline.
- **Direct MinIO/AWS S3 Access:** The library uses your MinIO or AWS credentials for all S3 operations. Make sure your `.env` or environment is set up correctly.

## Troubleshooting

- **Docker Compose errors:** Make sure Docker is running and your user has permission to run Docker commands.
- **Multi-user issues:** Each user gets their own temp/log directory. If you see permission errors, check directory ownership and permissions.

## Idea behind

This lib is developed using the same ideas and implementation as the [aws-deepracer-community/deepracer-for-cloud](https://github.com/aws-deepracer-community/deepracer-for-cloud) repo: _"A quick and easy way to get up and running with a DeepRacer training environment using a cloud virtual machine or a local computer"_.

---

**For more examples and advanced configuration, see the [examples/](examples/) directory.**
