import os
from typing import List


def _adjust_composes_file_names(composes_names: List[str]) -> List[str]:
    """
    Adjusts the names of Docker Compose files.

    Args:
        composes_names (List[str]): List of Docker Compose file names.

    Returns:
        List[str]: Adjusted list containing the paths to Docker Compose files.
    """
    flag = "-f"
    prefix = 'docker-compose-'
    suffix = '.yml'
    
    docker_composes_path = _discover_path_to_docker_composes()
    
    compose_files = []
    for compose_name in composes_names:
        compose_files.extend([docker_composes_path + prefix + compose_name + suffix])

    return compose_files


def _discover_path_to_docker_composes() -> str:
    """
    Discovers the absolute path to Docker Compose files.

    Returns:
        str: Full path to the directory containing Docker Compose files.
    """
    cwd = os.getcwd()

    root = cwd
    while root != os.path.dirname(root):
        config_path = os.path.join(root, "config", "drfc-images")
        if os.path.isdir(config_path):
            return config_path + os.sep
        root = os.path.dirname(root)

    raise FileNotFoundError("Could not locate 'config/drfc-images' directory from current path.")