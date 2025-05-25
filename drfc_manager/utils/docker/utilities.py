import os
from typing import List
from pathlib import Path

from drfc_manager.utils.paths import get_docker_compose_path


def _adjust_composes_file_names(composes_names: List[str]) -> List[str]:
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


def _discover_path_to_docker_composes() -> str:
    """
    Discovers the absolute path to Docker Compose files.

    Returns:
        str: Full path to the directory containing Docker Compose files.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    config_path = os.path.join(project_root, "config", "drfc-images")
    if os.path.isdir(config_path):
        return config_path + os.sep

    raise FileNotFoundError("Could not locate 'config/drfc-images' directory in the project root.")


def get_drfc_images_path():
    env_path = os.environ.get("DRFC_REPO_ABS_PATH")
    candidate_env = Path(env_path) / "config/drfc-images" if env_path else None
    if candidate_env and candidate_env.exists() and os.access(candidate_env, os.R_OK):
        print("Using ENV path:", candidate_env)
        return str(candidate_env)

    pkg_path = Path(__file__).parent.parent.parent
    candidate = pkg_path / "config/drfc-images"
    if candidate.exists():
        print("Using PKG path:", candidate)
        return str(candidate)

    cwd_candidate = Path.cwd() / "config/drfc-images"
    if cwd_candidate.exists():
        print("Using CWD path:", cwd_candidate)
        return str(cwd_candidate)
    raise FileNotFoundError("Could not locate 'config/drfc-images' directory from any known path.")