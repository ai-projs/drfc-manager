from typing import List

from drfc_manager.types.env_vars import EnvVars
from drfc_manager.utils.paths import get_docker_compose_path

env_vars = EnvVars()


def adjust_composes_file_names(composes_names: List[str]) -> List[str]:
    """
    Adjusts the names of Docker Compose files.

    Args:
        composes_names (List[str]): List of Docker Compose file names.

    Returns:
        List[str]: Adjusted list containing the paths to Docker Compose files.
    """
    compose_files = []
    for compose_name in composes_names:
        compose_path = get_docker_compose_path(compose_name)
        if compose_path.exists():
            compose_files.append(str(compose_path))
        else:
            raise FileNotFoundError(f"Docker compose file not found: {compose_path}")

    return compose_files
