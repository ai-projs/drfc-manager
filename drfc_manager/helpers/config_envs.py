import os
from typing import List

from dotenv import load_dotenv

from drfc_manager.types.config import ConfigEnvs


def _discover_path_to_config_envs() -> str | None:
  cwd = os.getcwd()

  root = cwd
  while root != os.path.dirname(root):
    config_path = os.path.join(root, "config")
    if os.path.isdir(config_path):
      return config_path + os.sep
    root = os.path.dirname(root)

  raise FileNotFoundError("Could not locate 'config/' directory from current path.")

def load_envs_from_files(paths: List[str]) -> None:
  """
  Loads environment variables from a list of .env file paths into the current process environment.

  Args:
      paths (List[str]): List of paths to .env files.

  Raises:
      FileNotFoundError: If any of the given .env files do not exist.
      Exception: If loading any file fails for an unknown reason.
  """
  for path in paths:
    if not os.path.exists(path):
      raise FileNotFoundError(f".env file not found: {path}")
    try:
      load_dotenv(dotenv_path=path, override=True)
    except Exception as e:
      raise Exception(f"Failed to load .env file: {path}") from e

def find_envs_files(envs_names: List[ConfigEnvs]) -> List[str]:
  envs_files = []

  envs_path = _discover_path_to_config_envs()
  suffix = '.env'

  for env_name in envs_names:
    envs_files.extend([envs_path + env_name + suffix])

  return envs_files