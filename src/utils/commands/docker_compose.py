from subprocess import run
from typing import List
from src.types.docker import DockerImages


import os


class DockerComposeCommands:
    """
    A class to manage Docker Compose commands.

    Attributes:
        _base_command (str): The base command for Docker Compose operations.

    Methods:
        __init__(): Initializes the DockerComposeCommands object with the base Docker Compose command.
        up(files_path: List[DockerImages]): Executes the 'docker-compose up' command with specified files.
        down(files_path: List[DockerImages]): Executes the 'docker-compose down' command.
    """
    
    def __init__(self):
        """
        Initializes the DockerComposeCommands object with the base Docker Compose command.
        """
        self._base_command = 'docker-compose'
    
    def up(self, files_path: List[DockerImages]):
        """
        Executes the 'docker-compose up' command with specified files.

        Args:
            files_path (List[DockerImages]): List of paths to the Docker Compose files.
        """
        try:
            composes = _adjust_composes_file_names(files_path)
            command = [self._base_command] + composes + ["up", "-d"]

            result = run(command, capture_output=True)
            if result.returncode != 0:
                raise Exception(result.stderr)
        except Exception as e:
            raise e
    
    def down(self, files_path: List[DockerImages]):
        """
        Executes the 'docker-compose down' command.
        
        Args:
            files_path (List[DockerImages]): List of paths to the Docker Compose files.
        """
        try:
            composes = _adjust_composes_file_names(files_path)
            command = [self._base_command] + composes + ["down"]
            
            result = run(command, capture_output=True)
            if result.returncode != 0:
                raise Exception(result.stderr)
        except Exception as e:
            raise e


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
        compose_files.extend([flag, docker_composes_path + prefix + compose_name + suffix])

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
