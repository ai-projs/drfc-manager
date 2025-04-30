import os

from typing import Dict, Any
from src.types.env_vars import EnvVars
from src.config import settings
from gloe import transformer

@transformer
def setup_evaluation_env(data: Dict[str, Any]):
    """
    Setup environment variables for evaluation using EnvVars dataclass
    and load them into the process environment.
    Reads necessary parameters (model_name, run_id, etc.) from the input data dictionary.
    """
    model_name = data.get('model_name')
    run_id = data.get('run_id')
    # clone = data.get('clone', False) # Not currently used by this function, but could be extracted
    # quiet = data.get('quiet', False) # Not currently used by this function, but could be extracted

    if model_name is None or run_id is None:
        raise ValueError("setup_evaluation_env requires 'model_name' and 'run_id' in the input data dictionary.")

    print(f"Setting up environment for evaluation run_id: {run_id}, model: {model_name}")
    
    env_vars = EnvVars(
        DR_RUN_ID=run_id,
        DR_LOCAL_S3_MODEL_PREFIX=model_name,
        DR_LOCAL_S3_BUCKET=settings.minio.bucket_name,
        DRFC_REPO_ABS_PATH=settings.docker.drfc_base_path,
    )
    
    try:
        env_vars.load_to_environment()
    except Exception as e:
        print(f"Warning: Failed to load DR_* vars into process environment: {e}")
        raise RuntimeError(f"Failed to set up environment: {e}") from e

    stack_name = f"deepracer-eval-{run_id}"
    os.environ["STACK_NAME"] = stack_name
    os.environ["ROBOMAKER_COMMAND"] = "./run.sh run evaluation.launch"
    os.environ["DR_CURRENT_PARAMS_FILE"] = os.getenv("DR_LOCAL_S3_EVAL_PARAMS_FILE", "eval_params.yaml")


    data["stack_name"] = stack_name
    if 'original_prefix' not in data:
         data['original_prefix'] = model_name

    return data