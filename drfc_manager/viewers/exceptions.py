class StreamProxyError(Exception):
    """Base exception for stream proxy errors."""

    pass


class ContainerConfigError(StreamProxyError):
    """Raised when there's an error in container configuration."""

    pass


class StreamConnectionError(StreamProxyError):
    """Raised when there's an error connecting to the stream."""

    pass


class StreamTimeoutError(StreamConnectionError):
    """Raised when stream connection times out."""

    pass


class StreamResponseError(StreamProxyError):
    """Raised when there's an error in stream response."""

    pass


class HealthCheckError(StreamProxyError):
    """Raised when health check fails."""

    pass


class StreamProxyRouteError(StreamProxyError):
    """Base exception for stream proxy route errors."""

    pass


class UnknownContainerError(StreamProxyRouteError):
    """Raised when an unknown container ID is requested."""

    pass


class StreamProxyHealthError(StreamProxyRouteError):
    """Raised when health check encounters an error."""

    pass


class StreamProxySocketError(StreamProxyHealthError):
    """Raised when socket connection fails during health check."""

    pass


class StreamProxyPingError(StreamProxyHealthError):
    """Raised when HTTP ping fails during health check."""

    pass
