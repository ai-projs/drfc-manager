import os
from typing import List

from python_on_whales import DockerClient

local_docker_daemon = os.getenv('LOCAL_SERVER_DOCKER_DAEMON')
remote_docker_daemon = os.getenv("REMOTE_SERVER_DOCKER_DAEMON")
base_url = remote_docker_daemon if remote_docker_daemon is not None else local_docker_daemon


class DockerComposeClient:
  _instance = None

  def __init__(self):
    raise RuntimeError("This is a Singleton class, invoke the get_instance() method instead")

  @classmethod
  def get_instance(cls, compose_files: List[str]):
    try:
      if cls._instance is None:
        cls._instance = DockerClient(host=base_url, compose_files=compose_files, compose_env_files=["/Users/jv/Desktop/uni/drfc-manager/src/utils/compose/.env"])
      return cls._instance
    except Exception as e:
      raise e