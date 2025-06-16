import os
import yaml
import tempfile
import subprocess
from typing import Dict, Any

from drfc_manager.config_env import settings
from drfc_manager.utils.docker.exceptions.base import DockerError
from drfc_manager.utils.logging import logger
from drfc_manager.utils.env_utils import get_subprocess_env
from drfc_manager.types.env_vars import EnvVars

env_vars = EnvVars()

class RedisManager:
    def __init__(self, config=settings):
        self.config = config

    def _run_command(
        self, command: list, check: bool = True
    ) -> subprocess.CompletedProcess:
        """Run a shell command and return the result."""
        try:
            env = get_subprocess_env(env_vars)

            result = subprocess.run(
                command, check=check, capture_output=True, text=True, env=env
            )
            return result
        except subprocess.CalledProcessError as e:
            raise DockerError(f"Command failed: {e.cmd}, Error: {e.stderr}") from e

    def add_redis_to_compose(self, compose_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add Redis service to Docker Compose configuration."""
        logger.info("Adding Redis service to Docker Compose configuration...")
        
        if "services" not in compose_data:
            compose_data["services"] = {}
            logger.info("Created services section in compose data")

        compose_data["services"]["redis"] = {
            "image": "redis:alpine",
            "restart": "always",
            "networks": ["default"]
        }
        logger.info("Added Redis service configuration")

        for service_name in ["rl_coach", "robomaker"]:
            if service_name in compose_data["services"]:
                logger.info(f"Configuring Redis environment for {service_name} service")
                service = compose_data["services"][service_name]

                if "environment" not in service:
                    service["environment"] = {}
                    logger.info(f"Created environment section for {service_name}")

                if isinstance(service["environment"], dict):
                    service["environment"].update({
                        "REDIS_HOST": "redis",
                        "REDIS_PORT": str(self.config.redis.port)
                    })
                    logger.info(f"Updated {service_name} environment with Redis configuration")
                elif isinstance(service["environment"], list):
                    service["environment"].extend([
                        "REDIS_HOST=redis",
                        f"REDIS_PORT={self.config.redis.port}"
                    ])
                    logger.info(f"Extended {service_name} environment list with Redis configuration")

                if "depends_on" not in service:
                    service["depends_on"] = ["redis"]
                    logger.info(f"Added Redis dependency for {service_name}")
                elif isinstance(service["depends_on"], list):
                    if "redis" not in service["depends_on"]:
                        service["depends_on"].append("redis")
                        logger.info(f"Added Redis to existing dependencies for {service_name}")

        if "version" in compose_data:
            del compose_data["version"]
            logger.info("Removed version from compose data")

        logger.info("Redis service configuration completed")
        return compose_data

    def create_modified_compose_file(self, training_compose_path: str) -> str:
        try:
            with open(training_compose_path, "r") as file:
                compose_data = yaml.safe_load(file)
        except Exception as e:
            raise DockerError(
                f"Failed to load base training compose file '{training_compose_path}': {e}"
            )

        temp_fd, temp_compose_path = tempfile.mkstemp(
            suffix=".yml", prefix="docker-compose-training-redis-"
        )
        os.close(temp_fd)

        modified_compose_data = self.add_redis_to_compose(compose_data)

        try:
            with open(temp_compose_path, "w") as file:
                yaml.dump(modified_compose_data, file)
            logger.info(
                f"Created modified compose file with Redis at {temp_compose_path}"
            )
            return temp_compose_path
        except Exception as e:
            os.remove(temp_compose_path)
            raise DockerError(
                f"Failed to write modified compose file '{temp_compose_path}': {e}"
            )
