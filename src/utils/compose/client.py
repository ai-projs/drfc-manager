import os
from typing import List

from python_on_whales import DockerClient

local_docker_daemon = os.getenv('LOCAL_SERVER_DOCKER_DAEMON')
remote_docker_daemon = os.getenv("REMOTE_SERVER_DOCKER_DAEMON")
base_url = remote_docker_daemon if remote_docker_daemon is not None else local_docker_daemon


class DockerComposeClient:
    def __init__(self, compose_files: List[str]):
        self.compose_files = compose_files
        # initialize the DockerClient with the provided compose files
        self.client = DockerClient(compose_files=compose_files)

    # Remove get_instance and simply instantiate when needed