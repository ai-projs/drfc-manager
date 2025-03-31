import os
from docker import APIClient


local_docker_daemon = os.getenv('LOCAL_SERVER_DOCKER_DAEMON')
remote_docker_daemon = os.getenv("REMOTE_SERVER_DOCKER_DAEMON")
base_url = remote_docker_daemon if remote_docker_daemon is not None else local_docker_daemon


class DockerClientServer:
    _instance = None
    
    def __init__(self):
        raise RuntimeError("This is a Singleton class, invoke the get_instance() method instead")
    
    @classmethod
    def get_instance(cls):
        try:
            if cls._instance is None:
                cls._instance = APIClient(base_url, use_ssh_client=True)
            return cls._instance
        except Exception as e:
            raise e
