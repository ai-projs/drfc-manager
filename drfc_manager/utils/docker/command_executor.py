import os
import subprocess
from typing import List, Optional
from contextlib import contextmanager

from drfc_manager.utils.docker.exceptions import DockerError
from drfc_manager.utils.logging_config import get_logger


class CommandExecutor:
    """Base class for executing Docker commands."""

    def __init__(self, env: Optional[dict] = None):
        """Initialize the command executor.

        Args:
            env: Optional environment variables to use when executing commands.
                 If None, uses the current environment.
        """
        self.env = env or os.environ.copy()
        self.logger = get_logger(__name__)

    @contextmanager
    def _managed_command_execution(self, command: List[str], check: bool = True):
        """Context manager for command execution.

        Args:
            command: List of command arguments to execute
            check: Whether to check the return code

        Yields:
            None

        Raises:
            DockerError: If the command fails and check is True
        """
        self.logger.debug("executing_command", command=" ".join(command))
        try:
            yield
        except subprocess.CalledProcessError as e:
            self.logger.error(
                "command_failed",
                command=" ".join(command),
                returncode=e.returncode,
                stderr=e.stderr,
            )
            raise DockerError(
                message=f"Docker command failed with exit code {e.returncode}",
                command=command,
                stderr=e.stderr,
            ) from e
        except Exception as e:
            self.logger.error(
                "command_execution_error", command=" ".join(command), error=str(e)
            )
            raise DockerError(
                message=f"Failed to execute command: {e}", command=command
            ) from e

    def run_command(
        self, command: List[str], check: bool = True, capture: bool = True
    ) -> subprocess.CompletedProcess:
        """Execute a command and return the result.

        Args:
            command: List of command arguments to execute
            check: Whether to check the return code
            capture: Whether to capture stdout and stderr

        Returns:
            CompletedProcess object containing the command result

        Raises:
            DockerError: If the command fails and check is True
        """
        with self._managed_command_execution(command, check):
            result = subprocess.run(
                command, check=check, capture_output=capture, text=True, env=self.env
            )

            if capture:
                if result.stdout:
                    self.logger.debug("command_stdout", output=result.stdout)
                if result.stderr:
                    self.logger.debug("command_stderr", output=result.stderr)

            return result
