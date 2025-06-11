from typing import List, Dict, Optional
from contextlib import contextmanager

from drfc_manager.utils.docker.command_executor import CommandExecutor
from drfc_manager.utils.docker.docker_constants import (
    DEFAULT_SWARM_OPTIONS,
    ERROR_MESSAGES,
)
from drfc_manager.types.docker import ComposeFileType
from drfc_manager.utils.docker.exceptions import SwarmError, ConfigError
from drfc_manager.utils.logging_config import get_logger
from drfc_manager.utils.paths import get_docker_compose_path


class SwarmManager:
    """Manages Docker Swarm operations."""

    def __init__(
        self, project_name: str, command_executor: Optional[CommandExecutor] = None
    ):
        """Initialize the Swarm manager.

        Args:
            project_name: Name of the Docker Swarm stack
            command_executor: Optional command executor instance
        """
        self.project_name = project_name
        self.command_executor = command_executor or CommandExecutor()
        self.logger = get_logger(__name__)

    @contextmanager
    def _managed_swarm_files(self, file_types: List[ComposeFileType]):
        """Context manager for handling swarm compose file paths.

        Args:
            file_types: List of compose file types to get paths for

        Yields:
            List of compose file paths

        Raises:
            ConfigError: If a compose file is not found
        """
        compose_files = []
        try:
            for file_type in file_types:
                compose_path = get_docker_compose_path(file_type.value)
                if compose_path.exists():
                    compose_files.append(str(compose_path))
                else:
                    self.logger.error(
                        "swarm_file_not_found",
                        path=str(compose_path),
                        file_type=file_type.value,
                    )
                    raise ConfigError(
                        ERROR_MESSAGES["compose_file_not_found"].format(compose_path),
                        config_key=file_type.value,
                    )
            yield compose_files
        except Exception as e:
            if not isinstance(e, ConfigError):
                self.logger.error("swarm_file_management_failed", error=str(e))
                raise ConfigError(
                    ERROR_MESSAGES["swarm_file_management_failed"].format(str(e))
                ) from e
            raise

    @contextmanager
    def _managed_swarm_operation(
        self, action: str, compose_files: List[str], options: Optional[Dict] = None
    ):
        """Context manager for handling swarm operations.

        Args:
            action: Action to perform (deploy, remove, etc.)
            compose_files: List of compose file paths
            options: Optional command options

        Yields:
            None

        Raises:
            SwarmError: If the swarm operation fails
        """
        try:
            cmd = self._build_swarm_command(compose_files, action, options)
            yield cmd
        except Exception as e:
            self.logger.error(f"{action}_operation_failed", error=str(e))
            raise SwarmError(
                ERROR_MESSAGES["command_failed"].format(str(e)),
                command=cmd if "cmd" in locals() else None,
            ) from e

    def _get_compose_file_paths(self, file_types: List[ComposeFileType]) -> List[str]:
        """Get full paths for swarm compose files.

        Args:
            file_types: List of compose file types to get paths for

        Returns:
            List of compose file paths

        Raises:
            ConfigError: If a compose file is not found
        """
        with self._managed_swarm_files(file_types) as compose_files:
            return compose_files

    def _build_swarm_command(
        self, compose_files: List[str], action: str, options: Optional[Dict] = None
    ) -> List[str]:
        """Build a Docker Swarm command.

        Args:
            compose_files: List of compose file paths
            action: Action to perform (deploy, remove, etc.)
            options: Optional command options

        Returns:
            List of command arguments

        Raises:
            ConfigError: If invalid options are provided
        """
        cmd = ["docker", "stack"]

        cmd.append(action)

        cmd.append(self.project_name)

        if action == "deploy":
            for file in compose_files:
                cmd.extend(["--compose-file", file])

            options = options or {}
            if options.get("prune", DEFAULT_SWARM_OPTIONS["prune"]):
                cmd.append("--prune")
            if options.get("resolve_image", DEFAULT_SWARM_OPTIONS["resolve_image"]):
                cmd.append("--resolve-image=always")

        return cmd

    def deploy(self, compose_files: List[str], options: Optional[Dict] = None) -> str:
        """Deploy a Docker Swarm stack.

        Args:
            compose_files: List of compose file paths
            options: Optional command options

        Returns:
            Command output

        Raises:
            SwarmError: If the deploy operation fails
            ResourceError: If resource allocation fails
        """
        with self._managed_swarm_operation("deploy", compose_files, options) as cmd:
            self.logger.info(
                "deploying_swarm_stack", project=self.project_name, files=compose_files
            )
            try:
                result = self.command_executor.run_command(cmd)
                return result.stdout
            except Exception as e:
                raise SwarmError(
                    ERROR_MESSAGES["deploy_failed"].format(str(e)), command=cmd
                ) from e

    def remove(self) -> str:
        """Remove a Docker Swarm stack.

        Returns:
            Command output

        Raises:
            SwarmError: If the remove operation fails
        """
        with self._managed_swarm_operation("remove", []) as cmd:
            self.logger.info("removing_swarm_stack", project=self.project_name)
            try:
                result = self.command_executor.run_command(cmd)
                return result.stdout
            except Exception as e:
                raise SwarmError(
                    ERROR_MESSAGES["remove_failed"].format(str(e)), command=cmd
                ) from e

    def ps(self) -> str:
        """List running services in the stack.

        Returns:
            Command output

        Raises:
            SwarmError: If the ps operation fails
        """
        cmd = ["docker", "stack", "ps", self.project_name]
        self.logger.debug("listing_swarm_services", project=self.project_name)
        try:
            result = self.command_executor.run_command(cmd, check=False)
            return result.stdout
        except Exception as e:
            raise SwarmError(
                ERROR_MESSAGES["ps_failed"].format(str(e)), command=cmd
            ) from e

    def services(self) -> str:
        """List services in the stack.

        Returns:
            Command output

        Raises:
            SwarmError: If the services operation fails
        """
        cmd = ["docker", "stack", "services", self.project_name]
        self.logger.debug("listing_swarm_stack_services", project=self.project_name)
        try:
            result = self.command_executor.run_command(cmd, check=False)
            return result.stdout
        except Exception as e:
            raise SwarmError(
                ERROR_MESSAGES["services_failed"].format(str(e)), command=cmd
            ) from e

    def check_stack_exists(self) -> bool:
        """Check if the stack exists.

        Returns:
            True if the stack exists, False otherwise

        Raises:
            SwarmError: If the check operation fails
        """
        cmd = ["docker", "stack", "ls", "--format", "{{.Name}}"]
        try:
            result = self.command_executor.run_command(cmd, check=False)
            exists = self.project_name in result.stdout.splitlines()
            self.logger.debug(
                "checking_stack_exists", stack=self.project_name, exists=exists
            )
            return exists
        except Exception as e:
            raise SwarmError(
                ERROR_MESSAGES["check_stack_failed"].format(str(e)), command=cmd
            ) from e

    def list_services(self) -> List[str]:
        """List services in the stack.

        Returns:
            List of service names

        Raises:
            SwarmError: If the list operation fails
        """
        cmd = [
            "docker",
            "stack",
            "ps",
            self.project_name,
            "--format",
            "{{.Name}}",
            "--filter",
            "desired-state=running",
        ]
        self.logger.debug("listing_swarm_services", stack=self.project_name)
        try:
            result = self.command_executor.run_command(cmd, check=False)
            if result.returncode == 0 and result.stdout:
                services = result.stdout.strip().splitlines()
                self.logger.debug(
                    "found_swarm_services", stack=self.project_name, count=len(services)
                )
                return services
            return []
        except Exception as e:
            raise SwarmError(
                ERROR_MESSAGES["list_services_failed"].format(str(e)), command=cmd
            ) from e
