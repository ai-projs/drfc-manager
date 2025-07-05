import socket
import time
from typing import List, Dict, Tuple
from drfc_manager.types.env_vars import EnvVars
import httpx
from fastapi import Request, Query, Response, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse

from drfc_manager.viewers.stream_proxy_utils import (
    parse_content_type,
    format_error_text,
    get_target_config,
    build_health_response,
    build_stream_url,
)
from drfc_manager.viewers.stream_proxy_handlers import (
    StreamClient,
    create_stream_generator,
    create_error_response,
    log_stream_request,
)
from drfc_manager.viewers.exceptions import (
    UnknownContainerError,
    StreamProxySocketError,
    StreamProxyPingError,
    StreamProxyHealthError,
)
from drfc_manager.utils.logging_config import get_logger
from drfc_manager.types.constants import (
    DEFAULT_QUALITY,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
    DEFAULT_TOPIC,
    HEALTH_CHECK_SOCKET_TIMEOUT,
    HEALTH_CHECK_PING_TIMEOUT,
    HTTPX_TIMEOUT_CONNECT,
    HTTPX_TIMEOUT_READ,
    HTTPX_STREAM_CHUNK_SIZE,
)

logger = get_logger(__name__)
env_vars = EnvVars()


def validate_container_id(container_id: str, containers: List[str]) -> None:
    """Validate container ID against known containers."""
    if containers and container_id not in containers:
        logger.warning(
            "unknown_container_id",
            container_id=container_id,
            known_containers=containers,
        )
        raise UnknownContainerError(f"Unknown container ID: {container_id}")


async def handle_stream_response(
    resp: httpx.Response,
    container_id: str,
    client_ip: str,
    target_url: str,
    start_time: float,
) -> Response:
    """Handle successful stream response."""
    elapsed = time.time() - start_time
    log_stream_request(container_id, client_ip, target_url, elapsed, resp.status_code)

    raw_ct = resp.headers.get("content-type")
    upstream_content_type, media_type = parse_content_type(raw_ct)

    logger.info(
        "streaming_started",
        container_id=container_id,
        content_type=upstream_content_type,
        media_type=media_type,
    )

    response_headers = {"Content-Type": upstream_content_type}
    return StreamingResponse(
        create_stream_generator(resp, container_id),
        media_type=media_type,
        headers=response_headers,
    )


async def handle_error_response(
    resp: httpx.Response,
    container_id: str,
    client_ip: str,
    target_url: str,
    start_time: float,
) -> Response:
    """Handle error stream response."""
    elapsed = time.time() - start_time
    log_stream_request(container_id, client_ip, target_url, elapsed, resp.status_code)

    error_text_bytes = await resp.aread()
    error_text = format_error_text(error_text_bytes)
    return create_error_response(
        status_code=502,
        message=f"Upstream Error {resp.status_code}",
        container_id=container_id,
        error_text=error_text,
    )


async def check_socket_connection(
    target_host: str, target_port: int
) -> Tuple[bool, str, Dict[str, str]]:
    """Check socket connection to target."""
    try:
        with socket.create_connection(
            (target_host, target_port), timeout=HEALTH_CHECK_SOCKET_TIMEOUT
        ):
            return True, "open", {}
    except socket.timeout:
        error_msg = f"Timeout connecting to {target_host}:{target_port} after {HEALTH_CHECK_SOCKET_TIMEOUT}s"
        logger.warning(
            "health_check_socket_timeout",
            target_host=target_host,
            target_port=target_port,
            timeout=HEALTH_CHECK_SOCKET_TIMEOUT,
        )
        raise StreamProxySocketError(error_msg)
    except socket.gaierror as e:
        error_msg = f"DNS lookup failed for {target_host}: {e}"
        logger.warning("health_check_dns_error", target_host=target_host, error=str(e))
        raise StreamProxySocketError(error_msg)
    except ConnectionRefusedError:
        error_msg = f"Connection refused by {target_host}:{target_port}"
        logger.warning(
            "health_check_connection_refused",
            target_host=target_host,
            target_port=target_port,
        )
        raise StreamProxySocketError(error_msg)
    except ConnectionResetError:
        error_msg = f"Connection reset by {target_host}:{target_port}"
        logger.warning(
            "health_check_connection_reset",
            target_host=target_host,
            target_port=target_port,
        )
        raise StreamProxySocketError(error_msg)
    except Exception as e:
        error_msg = (
            f"Unexpected error connecting to {target_host}:{target_port}: {str(e)}"
        )
        logger.warning(
            "health_check_socket_error",
            target_host=target_host,
            target_port=target_port,
            error_type=type(e).__name__,
            error=str(e),
            exc_info=True,
        )
        raise StreamProxySocketError(error_msg)


async def check_http_ping(
    client: httpx.AsyncClient, target_ping_url: str
) -> Tuple[bool, str, Dict[str, str]]:
    """Check HTTP ping to target."""
    try:
        resp = await client.get(target_ping_url)
        if 200 <= resp.status_code < 500:
            return True, f"ok (status: {resp.status_code})", {}
        else:
            error_msg = f"Target responded with server error status: {resp.status_code}"
            logger.warning(
                "health_check_bad_status",
                target_url=target_ping_url,
                status_code=resp.status_code,
            )
            raise StreamProxyPingError(error_msg)
    except httpx.TimeoutException:
        error_msg = f"Timeout connecting or reading from {target_ping_url} after {HEALTH_CHECK_PING_TIMEOUT}s"
        logger.warning(
            "health_check_ping_timeout",
            target_url=target_ping_url,
            timeout=HEALTH_CHECK_PING_TIMEOUT,
        )
        raise StreamProxyPingError(error_msg)
    except httpx.ConnectError:
        error_msg = f"HTTP Connection refused by {target_ping_url}"
        logger.warning("health_check_ping_refused", target_url=target_ping_url)
        raise StreamProxyPingError(error_msg)
    except httpx.RequestError as e:
        error_msg = f"HTTP request error for {target_ping_url}: {str(e)}"
        logger.warning(
            "health_check_request_error",
            target_url=target_ping_url,
            error_type=type(e).__name__,
            error=str(e),
        )
        raise StreamProxyPingError(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error during HTTP ping to {target_ping_url}: {str(e)}"
        logger.warning(
            "health_check_ping_error",
            target_url=target_ping_url,
            error_type=type(e).__name__,
            error=str(e),
            exc_info=True,
        )
        raise StreamProxyPingError(error_msg)


async def proxy_stream(
    request: Request,
    container_id: str,
    containers,
    topic: str = Query(DEFAULT_TOPIC, description="ROS topic to stream"),
    quality: int = Query(
        DEFAULT_QUALITY, description="Image quality (1-100)", ge=1, le=100
    ),
    width: int = Query(DEFAULT_WIDTH, description="Image width", ge=1),
    height: int = Query(DEFAULT_HEIGHT, description="Image height", ge=1),
):
    if containers and container_id not in containers:
        logger.warning(
            f"[{container_id}] Requested container_id not in known list configured via DR_VIEWER_CONTAINERS."
        )

    target_host = env_vars.DR_TARGET_HOST
    target_port = env_vars.DR_TARGET_PORT
    target_url = build_stream_url(
        target_host, target_port, topic, quality, width, height
    )

    client_ip = request.client.host if request.client else "unknown"
    logger.info(
        f"[{container_id}] Client '{client_ip}' requested stream. Proxying to: {target_url}"
    )

    client = httpx.AsyncClient(
        timeout=httpx.Timeout(HTTPX_TIMEOUT_READ, connect=HTTPX_TIMEOUT_CONNECT)
    )
    resp = None
    start_time = time.time()

    try:
        req = client.build_request("GET", target_url)
        resp = await client.send(req, stream=True)

        elapsed_connect = time.time() - start_time
        logger.info(
            f"[{container_id}] Upstream connection established in {elapsed_connect:.2f}s. Status: {resp.status_code}"
        )
        logger.debug(
            f"[{container_id}] Upstream Response Headers: {dict(resp.headers)}"
        )

        if resp.status_code == 200:
            upstream_content_type = resp.headers.get("content-type", "image/jpeg")
            if isinstance(upstream_content_type, bytes):
                upstream_content_type = upstream_content_type.decode("latin-1")
            media_type = upstream_content_type.split(";")[0].strip().lower()

            logger.info(
                f"[{container_id}] Streaming with Content-Type: '{upstream_content_type}' (Media Type: '{media_type}')"
            )

            response_headers = {"Content-Type": upstream_content_type}

            async def stream_generator():
                try:
                    async for chunk in resp.aiter_bytes(
                        chunk_size=HTTPX_STREAM_CHUNK_SIZE
                    ):
                        yield chunk
                except httpx.ReadError as stream_err:
                    logger.warning(
                        f"[{container_id}] Read error during stream iteration (client likely disconnected): {stream_err}"
                    )
                except Exception as stream_err:
                    logger.error(
                        f"[{container_id}] Unexpected error during stream iteration: {type(stream_err).__name__} - {stream_err}",
                        exc_info=True,
                    )
                finally:
                    logger.debug(
                        f"[{container_id}] Stream generator finished or terminated."
                    )

            async def close_resources():
                closed_resp = False
                closed_client = False
                try:
                    if resp and not resp.is_closed:
                        await resp.aclose()
                        closed_resp = True
                    if client and not client.is_closed:
                        await client.aclose()
                        closed_client = True
                except Exception as close_err:
                    logger.error(
                        f"[{container_id}] Error closing resources in background task: {close_err}",
                        exc_info=True,
                    )
                finally:
                    if closed_resp or closed_client:
                        logger.debug(
                            f"[{container_id}] Background task closed resources (Resp: {closed_resp}, Client: {closed_client})."
                        )

            bg_tasks = BackgroundTasks()
            bg_tasks.add_task(close_resources)
            return StreamingResponse(
                stream_generator(),
                media_type=media_type,
                headers=response_headers,
                background=bg_tasks,
            )

        else:
            error_text_bytes = await resp.aread()
            await resp.aclose()
            await client.aclose()
            error_text = error_text_bytes[:200].decode("utf-8", errors="replace")
            logger.error(
                f"[{container_id}] Upstream server error ({resp.status_code}): {error_text}"
            )
            return Response(
                content=f"Upstream Error {resp.status_code}",
                status_code=502,
                media_type="text/plain",
            )

    except httpx.TimeoutException as e:
        elapsed = time.time() - start_time
        logger.error(
            f"[{container_id}] Proxy Timeout connecting to upstream after {elapsed:.2f}s: {str(e)}"
        )
        if resp and not resp.is_closed:
            await resp.aclose()
        if client and not client.is_closed:
            await client.aclose()
        return Response(
            content="Proxy Timeout", status_code=504, media_type="text/plain"
        )
    except httpx.ConnectError as e:
        elapsed = time.time() - start_time
        logger.error(
            f"[{container_id}] Proxy Connection Error to upstream after {elapsed:.2f}s: {str(e)}"
        )
        if resp and not resp.is_closed:
            await resp.aclose()
        if client and not client.is_closed:
            await client.aclose()
        return Response(
            content="Proxy Connection Error", status_code=502, media_type="text/plain"
        )
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(
            f"[{container_id}] Unexpected error in proxy stream endpoint after {elapsed:.2f}s: {type(e).__name__} - {str(e)}",
            exc_info=True,
        )
        try:
            if resp and not resp.is_closed:
                await resp.aclose()
        except Exception as resp_close_err:
            logger.error(
                f"[{container_id}] Error closing response during exception handling: {resp_close_err}"
            )
        try:
            if client and not client.is_closed:
                await client.aclose()
        except Exception as client_close_err:
            logger.error(
                f"[{container_id}] Error closing client during exception handling: {client_close_err}"
            )
        return Response(
            content="Internal Proxy Error", status_code=500, media_type="text/plain"
        )


async def health_check(containers: List[str]) -> JSONResponse:
    """Handle health check request."""
    target_host, target_port = get_target_config()
    target_ping_url = f"http://{target_host}:{target_port}/"
    target_host = target_host if target_host else "localhost"

    try:
        target_reachable, socket_status, socket_errors = await check_socket_connection(
            target_host, target_port
        )

        if target_reachable:
            async with StreamClient() as (client, _):
                target_responsive, ping_status, ping_errors = await check_http_ping(
                    client, target_ping_url
                )
        else:
            target_responsive = False
            ping_status = "skipped (socket unreachable)"
            ping_errors = {}

        error_details = {**socket_errors, **ping_errors}
        response_data = build_health_response(
            target_host=target_host,
            target_port=target_port,
            socket_status=socket_status,
            ping_status=ping_status,
            containers=containers,
            error_details=error_details,
            target_reachable=target_reachable,
            target_responsive=target_responsive,
        )

        status_code = 200 if (target_reachable and target_responsive) else 503
        return JSONResponse(content=response_data, status_code=status_code)

    except StreamProxyHealthError as e:
        logger.error("health_check_failed", error=str(e))
        return JSONResponse(
            content={"status": "error", "message": str(e)}, status_code=503
        )
