from enum import Enum
from typing import Dict, Any


class DockerStyle(str, Enum):
    """Docker deployment styles."""

    COMPOSE = "compose"
    SWARM = "swarm"


# Default Docker Compose command options
DEFAULT_COMPOSE_OPTIONS: Dict[str, Any] = {
    "detach": True,
    "remove_orphans": True,
    "force_recreate": True,
}

# Default Docker Swarm command options
DEFAULT_SWARM_OPTIONS: Dict[str, Any] = {"detach": True}

# Environment variable names
ENV_VAR_NAMES = {
    "ROBOMAKER_GUI_PORT": "DR_ROBOMAKER_GUI_PORT",
    "ROBOMAKER_TRAIN_PORT": "DR_ROBOMAKER_TRAIN_PORT",
    "CURRENT_PARAMS_FILE": "DR_CURRENT_PARAMS_FILE",
    "RUN_ID": "DR_RUN_ID",
    "REDIS_HOST": "REDIS_HOST",
    "REDIS_PORT": "REDIS_PORT",
    "ROBOMAKER_COMMAND": "ROBOMAKER_COMMAND",
    "DOCKER_STYLE": "DR_DOCKER_STYLE",
}

# Default command paths
DEFAULT_COMMANDS = {
    "docker": "docker",
    "compose": "docker compose",
    "swarm": "docker stack",
}

# Error messages
ERROR_MESSAGES = {
    "compose_file_not_found": "Docker compose file not found: {}",
    "command_failed": "Docker command failed with exit code {}",
    "stack_exists": "Stack {} already running (found services). Stop evaluation first.",
    "network_not_found": "Could not locate 'config/drfc-images' directory in the project root.",
    "start_stack_failed": "Failed to start Docker stack: {}",
    "cleanup_failed": "Failed to cleanup previous run: {}",
    "deploy_failed": "Failed to deploy stack: {}",
    "remove_failed": "Failed to remove stack: {}",
    "ps_failed": "Failed to list services: {}",
    "services_failed": "Failed to list stack services: {}",
    "down_failed": "Failed to stop services: {}",
}
