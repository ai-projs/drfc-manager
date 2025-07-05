import json
from typing import List, Optional, Union, Tuple, Dict, Any

from drfc_manager.types.env_vars import EnvVars
from structlog import BoundLogger
from drfc_manager.viewers.exceptions import StreamResponseError
from drfc_manager.utils.logging_config import get_logger
from drfc_manager.types.constants import (
    DEFAULT_CONTENT_TYPE,
    DEFAULT_MEDIA_TYPE,
)

logger = get_logger(__name__)
env_vars = EnvVars()


def parse_containers(
    containers_str: Optional[str], logger: Optional[BoundLogger] = None
) -> List[str]:
    """Parse container IDs from environment variable."""
    if not containers_str:
        return []

    if logger:
        logger.info("parsing_containers", containers_str=containers_str)

    try:
        containers = json.loads(containers_str)
        if not isinstance(containers, list) or not all(
            isinstance(c, str) for c in containers
        ):
            if logger:
                logger.warning(
                    "invalid_container_config", containers_str=containers_str
                )
            return []

        if logger:
            logger.debug("containers_parsed", count=len(containers))
        return containers

    except json.JSONDecodeError:
        if logger:
            logger.warning("invalid_json", containers_str=containers_str)
        return []


def get_target_config(
    host: Optional[str] = None, port: Optional[int] = None
) -> Tuple[Optional[str], int]:
    """Get target host and port from environment or defaults."""
    target_host = host or env_vars.DR_TARGET_HOST
    target_port = port or env_vars.DR_DYNAMIC_PROXY_PORT

    logger.debug("target_config", host=target_host, port=target_port)
    return target_host, target_port


def build_stream_url(
    host: str, port: int, topic: str, quality: int, width: int, height: int
) -> str:
    return f"http://{host}:{port}/stream?topic={topic}&quality={quality}&width={width}&height={height}"


def parse_content_type(
    content_type: Optional[Union[str, bytes]], logger: Optional[BoundLogger] = None
) -> Tuple[str, str]:
    """Parse content type from response headers."""
    if not content_type:
        return DEFAULT_CONTENT_TYPE, DEFAULT_MEDIA_TYPE

    try:
        if isinstance(content_type, bytes):
            content_type = content_type.decode("utf-8")

        content_type = content_type.strip()
        if not content_type or "/" not in content_type:
            raise StreamResponseError("Invalid content type format")

        media_type = content_type.split(";")[0].strip()
        if logger:
            logger.debug(
                "content_type_parsed", content_type=content_type, media_type=media_type
            )
        return content_type, media_type

    except UnicodeDecodeError:
        raise StreamResponseError("Invalid content type encoding")
    except Exception as e:
        raise StreamResponseError(f"Error parsing content type: {e}")


def format_error_text(
    error_text: Optional[bytes],
    max_length: int = 200,
    logger: Optional[BoundLogger] = None,
) -> str:
    """Format error text from bytes to string."""
    if not error_text:
        return ""

    try:
        text = error_text.decode("utf-8")
        return text[:max_length] + "..." if len(text) > max_length else text
    except UnicodeDecodeError:
        if logger:
            logger.warning("invalid_encoding", error_text=error_text)
        return ""


def build_health_response(
    target_host: str,
    target_port: int,
    socket_status: str,
    ping_status: str,
    containers: List[str],
    error_details: Dict[str, Any],
    target_reachable: bool,
    target_responsive: bool,
) -> Dict[str, Any]:
    """Build health check response data."""
    healthy = target_reachable and target_responsive
    logger.info("health_check", status="healthy" if healthy else "unhealthy")

    response = {
        "status": "healthy" if healthy else "unhealthy",
        "details": {
            "proxy_type": "stream_proxy_httpx",
            "known_containers_count": len(containers),
            "target_stream_server": {
                "host": target_host,
                "port": target_port,
                "socket_check": socket_status,
                "http_check_url": f"http://{target_host}:{target_port}/",
                "http_check": ping_status,
            },
        },
    }

    if error_details:
        response["errors"] = error_details

    return response
