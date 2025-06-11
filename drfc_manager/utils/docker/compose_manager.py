from typing import List, Dict, Optional
from contextlib import contextmanager

from drfc_manager.types.docker import ComposeFileType
from drfc_manager.utils.docker.command_executor import CommandExecutor
from drfc_manager.utils.docker.docker_constants import (
    DEFAULT_COMPOSE_OPTIONS,
    ERROR_MESSAGES,
)
from drfc_manager.utils.docker.exceptions import (
    ComposeError,
    ContainerError,
    ConfigError,
    ResourceError,
)
from drfc_manager.utils.logging_config import get_logger
from drfc_manager.utils.paths import get_docker_compose_path


class ComposeManager:
    """Manages Docker Compose operations."""

    def __init__(
        self, project_name: str, command_executor: Optional[CommandExecutor] = None
    ):
        """Initialize the Compose manager.

        Args:
            project_name: Name of the Docker Compose project
            command_executor: Optional command executor instance
        """
        self.project_name = project_name
        self.command_executor = command_executor or CommandExecutor()
        self.logger = get_logger(__name__)

    @contextmanager
    def _managed_compose_files(self, file_types: List[ComposeFileType]):
        """Context manager for handling compose file paths.

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
                        "compose_file_not_found",
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
                self.logger.error("compose_file_management_failed", error=str(e))
                raise ConfigError(
                    ERROR_MESSAGES["compose_file_management_failed"].format(str(e))
                ) from e
            raise

    @contextmanager
    def _managed_compose_operation(
        self, action: str, compose_files: List[str], options: Optional[Dict] = None
    ):
        """Context manager for handling compose operations.

        Args:
            action: Action to perform (up, down, etc.)
            compose_files: List of compose file paths
            options: Optional command options

        Yields:
            None

        Raises:
            ComposeError: If the compose operation fails
        """
        try:
            cmd = self._build_compose_command(compose_files, action, options)
            yield cmd
        except Exception as e:
            self.logger.error(f"{action}_operation_failed", error=str(e))
            raise ComposeError(
                ERROR_MESSAGES["command_failed"].format(str(e)),
                command=cmd if "cmd" in locals() else None,
            ) from e

    def _get_compose_file_paths(self, file_types: List[ComposeFileType]) -> List[str]:
        """Get full paths for compose files.

        Args:
            file_types: List of compose file types to get paths for

        Returns:
            List of compose file paths

        Raises:
            ConfigError: If a compose file is not found
        """
        with self._managed_compose_files(file_types) as compose_files:
            return compose_files

    def _build_compose_command(
        self, compose_files: List[str], action: str, options: Optional[Dict] = None
    ) -> List[str]:
        """Build a Docker Compose command.

        Args:
            compose_files: List of compose file paths
            action: Action to perform (up, down, etc.)
            options: Optional command options

        Returns:
            List of command arguments

        Raises:
            ConfigError: If invalid options are provided
        """
        cmd = ["docker", "compose"]

        for file in compose_files:
            cmd.extend(["-f", file])

        cmd.extend(["-p", self.project_name])

        cmd.append(action)

        options = options or {}
        if action == "up":
            if options.get("detach", DEFAULT_COMPOSE_OPTIONS["detach"]):
                cmd.append("-d")
            if options.get("remove_orphans", DEFAULT_COMPOSE_OPTIONS["remove_orphans"]):
                cmd.append("--remove-orphans")
            if options.get("force_recreate", DEFAULT_COMPOSE_OPTIONS["force_recreate"]):
                cmd.append("--force-recreate")

        elif action == "down":
            if options.get("remove_volumes", True):
                cmd.append("--volumes")
            if options.get("remove_orphans", True):
                cmd.append("--remove-orphans")

        return cmd

    def up(
        self,
        compose_files: List[str],
        scale_options: Optional[Dict[str, int]] = None,
        options: Optional[Dict] = None,
    ) -> str:
        """Start Docker Compose services.

        Args:
            compose_files: List of compose file paths
            scale_options: Optional service scaling options
            options: Optional command options

        Returns:
            Command output

        Raises:
            ComposeError: If the up operation fails
            ResourceError: If scaling fails
        """
        with self._managed_compose_operation("up", compose_files, options) as cmd:
            if scale_options:
                for service, replicas in scale_options.items():
                    cmd.extend(["--scale", f"{service}={replicas}"])

            self.logger.info(
                "starting_compose_services",
                project=self.project_name,
                files=compose_files,
                scale_options=scale_options,
            )
            try:
                result = self.command_executor.run_command(cmd)
                return result.stdout
            except Exception as e:
                if scale_options:
                    raise ResourceError(
                        ERROR_MESSAGES["scaling_failed"].format(str(e)),
                        resource_type="service",
                        command=cmd,
                    ) from e
                raise ComposeError(
                    ERROR_MESSAGES["up_failed"].format(str(e)), command=cmd
                ) from e

    def down(self, compose_files: List[str], options: Optional[Dict] = None) -> str:
        """Stop Docker Compose services.

        Args:
            compose_files: List of compose file paths
            options: Optional command options

        Returns:
            Command output

        Raises:
            ComposeError: If the down operation fails
        """
        with self._managed_compose_operation("down", compose_files, options) as cmd:
            self.logger.info(
                "stopping_compose_services",
                project=self.project_name,
                files=compose_files,
            )
            try:
                result = self.command_executor.run_command(cmd)
                return result.stdout
            except Exception as e:
                raise ComposeError(
                    ERROR_MESSAGES["down_failed"].format(str(e)), command=cmd
                ) from e

    def ps(self) -> str:
        """List running services.

        Returns:
            Command output

        Raises:
            ComposeError: If the ps operation fails
        """
        cmd = ["docker", "compose", "-p", self.project_name, "ps"]
        self.logger.debug("listing_compose_services", project=self.project_name)
        try:
            result = self.command_executor.run_command(cmd, check=False)
            return result.stdout
        except Exception as e:
            raise ComposeError(
                ERROR_MESSAGES["ps_failed"].format(str(e)), command=cmd
            ) from e

    def logs(self, service_name: str, tail: int = 30) -> str:
        """Get logs for a service.

        Args:
            service_name: Name of the service
            tail: Number of lines to show

        Returns:
            Command output

        Raises:
            ContainerError: If the logs operation fails
        """
        cmd = [
            "docker",
            "compose",
            "-p",
            self.project_name,
            "logs",
            service_name,
            "--tail",
            str(tail),
        ]
        self.logger.debug(
            "getting_service_logs",
            project=self.project_name,
            service=service_name,
            tail=tail,
        )
        try:
            result = self.command_executor.run_command(cmd, check=False)
            return result.stdout
        except Exception as e:
            raise ContainerError(
                ERROR_MESSAGES["logs_failed"].format(str(e)),
                container_id=service_name,
                command=cmd,
            ) from e
