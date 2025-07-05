import os
from typing import Dict
from dataclasses import asdict
from drfc_manager.types.env_vars import EnvVars
from drfc_manager.utils.logging_config import get_logger

logger = get_logger(__name__)


def get_subprocess_env(env_vars: EnvVars) -> Dict[str, str]:
    """
    Creates a copy of the current environment and updates it with values from EnvVars.
    This ensures that subprocess commands have access to all necessary environment variables.

    Args:
        env_vars: The EnvVars instance containing the environment variables

    Returns:
        Dict[str, str]: A copy of the environment with updated variables
    """
    env = os.environ.copy()
    
    env_dict = {
        k: str(v)
        for k, v in asdict(env_vars).items()
        if not k.startswith("_") and v is not None
    }
    
    env.update(env_dict)
    
    return env
