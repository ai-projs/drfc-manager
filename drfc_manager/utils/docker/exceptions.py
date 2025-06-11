from typing import List, Optional


class DockerError(Exception):
    """Custom exception for Docker-related errors."""

    def __init__(
        self,
        message: str,
        command: Optional[List[str]] = None,
        stderr: Optional[str] = None,
    ):
        super().__init__(message)
        self.command = command
        self.stderr = stderr

    def __str__(self):
        msg = super().__str__()
        if self.command:
            msg += f"\nCommand: {' '.join(self.command)}"
        if self.stderr:
            msg += f"\nStderr:\n{self.stderr}"
        return msg


class ComposeError(DockerError):
    """Exception raised for Docker Compose related errors."""

    pass


class SwarmError(DockerError):
    """Exception raised for Docker Swarm related errors."""

    pass


class ContainerError(DockerError):
    """Exception raised for container-related errors."""

    def __init__(
        self,
        message: str,
        container_id: Optional[str] = None,
        command: Optional[List[str]] = None,
        stderr: Optional[str] = None,
    ):
        super().__init__(message, command, stderr)
        self.container_id = container_id

    def __str__(self):
        msg = super().__str__()
        if self.container_id:
            msg += f"\nContainer ID: {self.container_id}"
        return msg


class ImageError(DockerError):
    """Exception raised for Docker image related errors."""

    def __init__(
        self,
        message: str,
        image_name: Optional[str] = None,
        command: Optional[List[str]] = None,
        stderr: Optional[str] = None,
    ):
        super().__init__(message, command, stderr)
        self.image_name = image_name

    def __str__(self):
        msg = super().__str__()
        if self.image_name:
            msg += f"\nImage: {self.image_name}"
        return msg


class NetworkError(DockerError):
    """Exception raised for Docker network related errors."""

    def __init__(
        self,
        message: str,
        network_name: Optional[str] = None,
        command: Optional[List[str]] = None,
        stderr: Optional[str] = None,
    ):
        super().__init__(message, command, stderr)
        self.network_name = network_name

    def __str__(self):
        msg = super().__str__()
        if self.network_name:
            msg += f"\nNetwork: {self.network_name}"
        return msg


class VolumeError(DockerError):
    """Exception raised for Docker volume related errors."""

    def __init__(
        self,
        message: str,
        volume_name: Optional[str] = None,
        command: Optional[List[str]] = None,
        stderr: Optional[str] = None,
    ):
        super().__init__(message, command, stderr)
        self.volume_name = volume_name

    def __str__(self):
        msg = super().__str__()
        if self.volume_name:
            msg += f"\nVolume: {self.volume_name}"
        return msg


class ConfigError(DockerError):
    """Exception raised for Docker configuration related errors."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        command: Optional[List[str]] = None,
        stderr: Optional[str] = None,
    ):
        super().__init__(message, command, stderr)
        self.config_key = config_key

    def __str__(self):
        msg = super().__str__()
        if self.config_key:
            msg += f"\nConfig Key: {self.config_key}"
        return msg


class ResourceError(DockerError):
    """Exception raised for Docker resource related errors."""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        command: Optional[List[str]] = None,
        stderr: Optional[str] = None,
    ):
        super().__init__(message, command, stderr)
        self.resource_type = resource_type
        self.resource_id = resource_id

    def __str__(self):
        msg = super().__str__()
        if self.resource_type:
            msg += f"\nResource Type: {self.resource_type}"
        if self.resource_id:
            msg += f"\nResource ID: {self.resource_id}"
        return msg
