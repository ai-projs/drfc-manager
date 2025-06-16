import time
from typing import Optional, AsyncGenerator, Tuple
import httpx
from fastapi import Response
from drfc_manager.utils.logging_config import get_logger
from drfc_manager.types.constants import (
    HTTPX_TIMEOUT_CONNECT,
    HTTPX_TIMEOUT_READ,
    HTTPX_STREAM_CHUNK_SIZE,
)

logger = get_logger(__name__)


class StreamClient:
    """Context manager for handling stream client and response."""

    def __init__(
        self,
        timeout_read: float = HTTPX_TIMEOUT_READ,
        timeout_connect: float = HTTPX_TIMEOUT_CONNECT,
    ):
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_read, connect=timeout_connect)
        )
        self.response: Optional[httpx.Response] = None
        self.start_time = time.time()

    async def __aenter__(self) -> Tuple[httpx.AsyncClient, Optional[httpx.Response]]:
        return self.client, self.response

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.response and not self.response.is_closed:
                await self.response.aclose()
        except Exception as e:
            logger.error("response_close_error", error=str(e))

        try:
            if self.client and not self.client.is_closed:
                await self.client.aclose()
        except Exception as e:
            logger.error("client_close_error", error=str(e))


async def create_stream_generator(
    response: httpx.Response,
    container_id: str,
    chunk_size: int = HTTPX_STREAM_CHUNK_SIZE,
) -> AsyncGenerator[bytes, None]:
    """
    Create a generator for streaming response chunks.

    Args:
        response: HTTPX response object
        container_id: Container ID for logging
        chunk_size: Size of chunks to stream

    Yields:
        Response chunks
    """
    try:
        async for chunk in response.aiter_bytes(chunk_size=chunk_size):
            yield chunk
    except httpx.ReadError as stream_err:
        logger.warning(
            "stream_read_error", container_id=container_id, error=str(stream_err)
        )
    except Exception as stream_err:
        logger.error(
            "stream_unexpected_error",
            container_id=container_id,
            error_type=type(stream_err).__name__,
            error=str(stream_err),
            exc_info=True,
        )
    finally:
        logger.debug("stream_generator_finished", container_id=container_id)


def create_error_response(
    status_code: int, message: str, container_id: str, error_text: Optional[str] = None
) -> Response:
    """
    Create a standardized error response.

    Args:
        status_code: HTTP status code
        message: Error message
        container_id: Container ID for logging
        error_text: Optional detailed error text

    Returns:
        FastAPI Response object
    """
    logger.error(
        "proxy_error",
        container_id=container_id,
        status_code=status_code,
        message=message,
        error_text=error_text,
    )

    return Response(content=message, status_code=status_code, media_type="text/plain")


def log_stream_request(
    container_id: str,
    client_ip: str,
    target_url: str,
    elapsed: float,
    status_code: Optional[int] = None,
) -> None:
    """
    Log stream request details.

    Args:
        container_id: Container ID
        client_ip: Client IP address
        target_url: Target URL
        elapsed: Time elapsed
        status_code: Optional status code
    """
    if status_code is not None:
        logger.info(
            "stream_request_complete",
            container_id=container_id,
            client_ip=client_ip,
            target_url=target_url,
            elapsed_seconds=elapsed,
            status_code=status_code,
        )
    else:
        logger.info(
            "stream_request_start",
            container_id=container_id,
            client_ip=client_ip,
            target_url=target_url,
        )
