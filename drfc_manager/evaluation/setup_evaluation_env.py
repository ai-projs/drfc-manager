import os
import datetime

from typing import Dict, Any
from src.types.env_vars import EnvVars
from src.config import settings
from gloe import transformer
from src.utils.logging import logger

@transformer
def setup_evaluation_env(data: Dict[str, Any]):
    """
    Setup environment variables for evaluation using EvaluationConfig
    and load them into the process environment.
    Reads necessary parameters from the config in the input data dictionary.
    """
    config = data.get('config')
    if not config:
        raise ValueError("setup_evaluation_env requires 'config' in the input data dictionary.")

    logger.info(f"Setting up environment for evaluation run_id: {config.run_id}, model: {config.model_name}")
    
    # Apply any overrides to the env_vars
    config.apply_overrides()
    
    # Set evaluation-specific ports with offsets based on run_id
    base_webviewer_port = 8100  # This should match your base configuration
    base_robomaker_port = 8080  # This should match your base configuration
    base_gui_port = 5900       # This should match your base configuration
    
    port_offset = config.run_id * 10  # Give each run_id a range of 10 ports
    
    # Set the ports with offsets
    os.environ["DR_WEBVIEWER_PORT"] = str(base_webviewer_port + port_offset)
    os.environ["DR_ROBOMAKER_EVAL_PORT"] = str(base_robomaker_port + port_offset)
    os.environ["DR_ROBOMAKER_GUI_PORT"] = str(base_gui_port + port_offset)
    
    try:
        config.env_vars.load_to_environment()
    except Exception as e:
        logger.warning(f"Failed to load DR_* vars into process environment: {e}")
        raise RuntimeError(f"Failed to set up environment: {e}") from e

    stack_name = f"deepracer-eval-{config.run_id}"
    os.environ["STACK_NAME"] = stack_name
    os.environ["ROBOMAKER_COMMAND"] = "./run.sh run evaluation.launch"
    os.environ["DR_CURRENT_PARAMS_FILE"] = os.getenv("DR_LOCAL_S3_EVAL_PARAMS_FILE", "eval_params.yaml")

    # Pass through all necessary data
    data.update({
        "stack_name": stack_name,
        "model_name": config.model_name,
        "original_prefix": config.model_name,
        "clone": config.clone,
        "quiet": config.quiet
    })

    eval_time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    config.env_vars.DR_SIMTRACE_S3_PREFIX = f'{config.model_name}/evaluation-{eval_time}'
    data["run_timestamp"] = eval_time

    return data