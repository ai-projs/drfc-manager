from drfc_manager.utils.logging_config import get_logger
import yaml
import tempfile
import subprocess

from drfc_manager.config_env import settings
from drfc_manager.utils.docker.exceptions import DockerError
from drfc_manager.utils.logging import logger
from drfc_manager.utils.docker.docker_constants import DockerStyle


class RedisManager:
    def __init__(
        self, config=settings, docker_style: DockerStyle = DockerStyle.COMPOSE
    ):
        self.config = config
        self.docker_style = docker_style
        self.logger = get_logger(__name__)

    def _run_command(
        self, command: list, check: bool = True
    ) -> subprocess.CompletedProcess:
        """Run a shell command and return the result."""
        try:
            result = subprocess.run(
                command, check=check, capture_output=True, text=True
            )
            return result
        except subprocess.CalledProcessError as e:
            raise DockerError(f"Command failed: {e.cmd}, Error: {e.stderr}") from e

    def ensure_network_exists(self) -> None:
        """Ensure the sagemaker-local network exists by removing any existing one and creating a fresh one."""
        network_name = "sagemaker-local"
        try:
            # First, remove the network if it exists
            self._run_command(
                ["docker", "network", "rm", network_name],
                check=False,  # Don't check since it might not exist
            )
            self.logger.info(f"Removed existing {network_name} network if it existed")

            # Create a fresh network
            create_command = ["docker", "network", "create"]
            if self.docker_style == DockerStyle.SWARM:
                # For Swarm, create an attachable overlay network
                swarm_info_result = self._run_command(
                    ["docker", "info", "--format", "{{.Swarm.LocalNodeState}}"],
                    check=False,
                )
                swarm_state = swarm_info_result.stdout.strip()
                if swarm_state != "active":
                    self.logger.warning(
                        "Docker Swarm mode is not active. Attempting to create an overlay network "
                        f"for {network_name} might fail or not behave as expected. "
                        "Please ensure Swarm is initialized if DR_DOCKER_STYLE is Swarm."
                    )
                create_command.extend(
                    ["--driver", "overlay", "--attachable", network_name]
                )
            else:  # Default to bridge for COMPOSE style
                create_command.append(network_name)

            self._run_command(create_command)
            self.logger.info(
                f"Created fresh {network_name} network with driver {'overlay' if self.docker_style == DockerStyle.SWARM else 'bridge'}"
            )
        except DockerError as e:
            self.logger.error(
                "failed_to_ensure_network_exists",
                network_name=network_name,
                error=str(e),
            )
            raise
        except Exception as e:
            self.logger.error(
                "unexpected_error_ensuring_network",
                network_name=network_name,
                error=str(e),
            )
            raise DockerError(
                f"Unexpected error ensuring {network_name} network exists: {str(e)}"
            ) from e

    def create_modified_compose_file(self, base_compose_path: str) -> str:
        """Create a temporary modified compose file.

        Args:
            base_compose_path: Path to the base compose file

        Returns:
            Path to the temporary compose file
        """
        try:
            # Ensure network exists
            self.ensure_network_exists()

            # Read base compose file
            with open(base_compose_path, "r") as f:
                compose_data = yaml.safe_load(f)

            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".yml", delete=False
            )

            # Write modified compose data
            yaml.dump(compose_data, temp_file)
            temp_file.close()

            return temp_file.name
        except Exception as e:
            logger.error(f"Failed to create modified compose file: {str(e)}")
            raise
