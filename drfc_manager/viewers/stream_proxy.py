import httpx
import uvicorn
import os
import json
import logging
import time
import socket
import tempfile
from typing import Dict, List, Optional

from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.responses import StreamingResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

user_tmp = os.path.join(tempfile.gettempdir(), os.environ.get('USER', 'unknown_user'))
try:
    os.makedirs(user_tmp, exist_ok=True)
except Exception as e:
    raise RuntimeError(f"Could not create user temp directory {user_tmp}: {e}")

LOG_FILE = os.path.join(user_tmp, "stream_proxy.log")
PROXY_STDOUT_LOG = os.path.join(user_tmp, "stream_proxy_8090_stdout.log")
PROXY_STDERR_LOG = os.path.join(user_tmp, "stream_proxy_8090_stderr.log")
DEFAULT_PROXY_PORT = 8090
DEFAULT_TARGET_HOST = "localhost"
DEFAULT_TARGET_PORT = 9080
DEFAULT_QUALITY = 75
DEFAULT_WIDTH = 480
DEFAULT_HEIGHT = 360
DEFAULT_TOPIC = "/racecar/deepracer/kvs_stream"
HTTPX_TIMEOUT_CONNECT = 10.0
HTTPX_TIMEOUT_READ = 30.0
HTTPX_STREAM_CHUNK_SIZE = 65536
HEALTH_CHECK_SOCKET_TIMEOUT = 2.0
HEALTH_CHECK_PING_TIMEOUT = 5.0

log_formatter = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')

# Only add console handler if explicitly requested
console_logging = os.environ.get('DRFC_CONSOLE_LOGGING', 'false').lower() == 'true'
handlers = []

if console_logging:
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    handlers.append(stream_handler)

try:
    file_handler = logging.FileHandler(LOG_FILE, mode='a')
    file_handler.setFormatter(log_formatter)
    handlers.append(file_handler)
except OSError as e:
    if not console_logging:
        # Add stream handler as fallback only if no console logging was configured
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(log_formatter)
        handlers.append(stream_handler)

logger = logging.getLogger('stream_proxy')
logger.setLevel(logging.INFO)
for handler in handlers:
    logger.addHandler(handler)
logger.propagate = False

if len(handlers) == 1 and isinstance(handlers[0], logging.StreamHandler) and not console_logging:
    logger.warning(f"Could not open log file {LOG_FILE}. Logging to console only.")

app = FastAPI(title="DeepRacer Stream Proxy")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

containers: List[str] = []
containers_str = os.environ.get('DR_VIEWER_CONTAINERS', '[]')
try:
    parsed_containers = json.loads(containers_str)
    if isinstance(parsed_containers, list) and all(isinstance(item, str) for item in parsed_containers):
        containers = parsed_containers
        logger.info(f"Successfully loaded {len(containers)} container IDs from DR_VIEWER_CONTAINERS.")
        logger.debug(f"Loaded container IDs: {containers}")
    else:
        logger.warning(f"DR_VIEWER_CONTAINERS was not a list of strings: '{containers_str}'. Treating as empty list.")
except json.JSONDecodeError:
    logger.error(f"Failed to parse DR_VIEWER_CONTAINERS JSON: '{containers_str}'. Treating as empty list.")
except Exception as e:
     logger.error(f"Unexpected error loading DR_VIEWER_CONTAINERS: {e}", exc_info=True)


@app.get("/{container_id}/stream")
async def proxy_stream(
    request: Request,
    container_id: str,
    topic: str = Query(DEFAULT_TOPIC, description="ROS topic to stream"),
    quality: int = Query(DEFAULT_QUALITY, description="Image quality (1-100)", ge=1, le=100),
    width: int = Query(DEFAULT_WIDTH, description="Image width", ge=1),
    height: int = Query(DEFAULT_HEIGHT, description="Image height", ge=1)
):
    if containers and container_id not in containers:
         logger.warning(f"[{container_id}] Requested container_id not in known list configured via DR_VIEWER_CONTAINERS.")

    target_host = os.environ.get('DR_TARGET_HOST', DEFAULT_TARGET_HOST)
    target_port = int(os.environ.get('DR_TARGET_PORT', DEFAULT_TARGET_PORT))
    target_url = f"http://{target_host}:{target_port}/stream?topic={topic}&quality={quality}&width={width}&height={height}"

    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"[{container_id}] Client '{client_ip}' requested stream. Proxying to: {target_url}")

    client = httpx.AsyncClient(timeout=httpx.Timeout(HTTPX_TIMEOUT_READ, connect=HTTPX_TIMEOUT_CONNECT))
    resp: Optional[httpx.Response] = None
    start_time = time.time()

    try:
        req = client.build_request("GET", target_url)
        resp = await client.send(req, stream=True)

        elapsed_connect = time.time() - start_time
        logger.info(f"[{container_id}] Upstream connection established in {elapsed_connect:.2f}s. Status: {resp.status_code}")
        logger.debug(f"[{container_id}] Upstream Response Headers: {dict(resp.headers)}")

        if resp.status_code == 200:
            upstream_content_type = resp.headers.get('content-type', 'image/jpeg')
            if isinstance(upstream_content_type, bytes):
                upstream_content_type = upstream_content_type.decode('latin-1')
            media_type = upstream_content_type.split(';')[0].strip().lower()

            logger.info(f"[{container_id}] Streaming with Content-Type: '{upstream_content_type}' (Media Type: '{media_type}')")

            response_headers = {'Content-Type': upstream_content_type}

            async def stream_generator():
                try:
                    async for chunk in resp.aiter_bytes(chunk_size=HTTPX_STREAM_CHUNK_SIZE):
                        yield chunk
                except httpx.ReadError as stream_err:
                    logger.warning(f"[{container_id}] Read error during stream iteration (client likely disconnected): {stream_err}")
                except Exception as stream_err:
                    logger.error(f"[{container_id}] Unexpected error during stream iteration: {type(stream_err).__name__} - {stream_err}", exc_info=True)
                finally:
                     logger.debug(f"[{container_id}] Stream generator finished or terminated.")

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
                      logger.error(f"[{container_id}] Error closing resources in background task: {close_err}", exc_info=True)
                 finally:
                      if closed_resp or closed_client:
                           logger.debug(f"[{container_id}] Background task closed resources (Resp: {closed_resp}, Client: {closed_client}).")


            return StreamingResponse(
                stream_generator(),
                media_type=media_type,
                headers=response_headers,
                background=close_resources
            )

        else:
            error_text_bytes = await resp.aread()
            await resp.aclose()
            await client.aclose()
            error_text = error_text_bytes[:200].decode('utf-8', errors='replace')
            logger.error(f"[{container_id}] Upstream server error ({resp.status_code}): {error_text}")
            return Response(content=f"Upstream Error {resp.status_code}", status_code=502, media_type="text/plain")

    except httpx.TimeoutException as e:
        elapsed = time.time() - start_time
        logger.error(f"[{container_id}] Proxy Timeout connecting to upstream after {elapsed:.2f}s: {str(e)}")
        if resp and not resp.is_closed: await resp.aclose()
        if client and not client.is_closed: await client.aclose()
        return Response(content="Proxy Timeout", status_code=504, media_type="text/plain")
    except httpx.ConnectError as e:
        elapsed = time.time() - start_time
        logger.error(f"[{container_id}] Proxy Connection Error to upstream after {elapsed:.2f}s: {str(e)}")
        if resp and not resp.is_closed: await resp.aclose()
        if client and not client.is_closed: await client.aclose()
        return Response(content="Proxy Connection Error", status_code=502, media_type="text/plain")
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[{container_id}] Unexpected error in proxy stream endpoint after {elapsed:.2f}s: {type(e).__name__} - {str(e)}", exc_info=True)
        try:
            if resp and not resp.is_closed: await resp.aclose()
        except Exception as resp_close_err:
            logger.error(f"[{container_id}] Error closing response during exception handling: {resp_close_err}")
        try:
            if client and not client.is_closed: await client.aclose()
        except Exception as client_close_err:
             logger.error(f"[{container_id}] Error closing client during exception handling: {client_close_err}")
        return Response(content="Internal Proxy Error", status_code=500, media_type="text/plain")


@app.get("/health")
async def health():
    target_host = os.environ.get('DR_TARGET_HOST', DEFAULT_TARGET_HOST)
    target_port = int(os.environ.get('DR_TARGET_PORT', DEFAULT_TARGET_PORT))
    target_ping_url = f"http://{target_host}:{target_port}/"

    ping_status = "unchecked"
    socket_status = "unchecked"
    target_reachable = False
    target_responsive = False
    error_details = {}

    try:
        with socket.create_connection((target_host, target_port), timeout=HEALTH_CHECK_SOCKET_TIMEOUT) as sock:
            socket_status = "open"
            target_reachable = True
    except socket.timeout:
        socket_status = "error: timeout"
        error_details["socket"] = f"Timeout connecting to {target_host}:{target_port} after {HEALTH_CHECK_SOCKET_TIMEOUT}s"
        logger.warning(f"Health check: Socket connection to {target_host}:{target_port} timed out.")
    except socket.gaierror as e:
        socket_status = "error: DNS lookup failed"
        error_details["socket"] = f"DNS lookup failed for {target_host}: {e}"
        logger.warning(f"Health check: DNS lookup failed for {target_host}: {e}")
    except ConnectionRefusedError:
        socket_status = "error: connection refused"
        error_details["socket"] = f"Connection refused by {target_host}:{target_port}"
        logger.warning(f"Health check: Connection refused by {target_host}:{target_port}.")
    except Exception as e:
        socket_status = f"error: {type(e).__name__}"
        error_details["socket"] = str(e)
        logger.warning(f"Health check: Socket connection test to {target_host}:{target_port} failed: {e}", exc_info=True)

    if target_reachable:
        ping_status = "checking"
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(HEALTH_CHECK_PING_TIMEOUT, connect=HEALTH_CHECK_SOCKET_TIMEOUT)) as client:
                resp = await client.get(target_ping_url)
                ping_status = f"ok (status: {resp.status_code})"
                if 200 <= resp.status_code < 500:
                     target_responsive = True
                else:
                     error_details["ping"] = f"Target responded with server error status: {resp.status_code}"
                     logger.warning(f"Health check: Target {target_ping_url} responded with status {resp.status_code}")

        except httpx.TimeoutException:
             ping_status = "error: timeout"
             error_details["ping"] = f"Timeout connecting or reading from {target_ping_url} after {HEALTH_CHECK_PING_TIMEOUT}s"
             logger.warning(f"Health check: HTTP ping to {target_ping_url} timed out.")
        except httpx.ConnectError:
             ping_status = "error: connection refused"
             error_details["ping"] = f"HTTP Connection refused by {target_ping_url}"
             logger.warning(f"Health check: HTTP connection refused by {target_ping_url}.")
        except httpx.RequestError as e:
             ping_status = f"error: {type(e).__name__}"
             error_details["ping"] = f"HTTP request error for {target_ping_url}: {str(e)}"
             logger.warning(f"Health check: HTTP request error for {target_ping_url}: {e}")
        except Exception as e:
             ping_status = f"error: {type(e).__name__}"
             error_details["ping"] = f"Unexpected error during HTTP ping to {target_ping_url}: {str(e)}"
             logger.warning(f"Health check: Unexpected error during HTTP ping to {target_ping_url}: {e}", exc_info=True)
    else:
        ping_status = "skipped (socket unreachable)"

    healthy = target_reachable and target_responsive

    response_data = {
        "status": "healthy" if healthy else "unhealthy",
        "details": {
            "proxy_type": "stream_proxy_httpx",
            "known_containers_count": len(containers),
            "target_stream_server": {
                "host": target_host,
                "port": target_port,
                "socket_check": socket_status,
                "http_check_url": target_ping_url,
                "http_check": ping_status,
            }
        }
    }
    if error_details:
         response_data["errors"] = error_details

    status_code = 200 if healthy else 503
    return JSONResponse(content=response_data, status_code=status_code)


if __name__ == "__main__":
    port = int(os.environ.get('DR_PROXY_PORT', DEFAULT_PROXY_PORT))
    host = "0.0.0.0"

    target_host = os.environ.get('DR_TARGET_HOST', DEFAULT_TARGET_HOST)
    target_port = int(os.environ.get('DR_TARGET_PORT', DEFAULT_TARGET_PORT))

    logger.info(f"Starting DeepRacer Stream Proxy Server on {host}:{port}")
    logger.info(f"Proxying streams to target: http://{target_host}:{target_port}")
    logger.info(f"Logging to console and file: {LOG_FILE}")
    if containers:
        logger.info(f"Aware of {len(containers)} container IDs: {containers}")
    else:
        logger.info("No specific container IDs loaded (DR_VIEWER_CONTAINERS not set or empty/invalid). Proxying requests for any container ID.")

    uvicorn.run(
        "stream_proxy:app",
        host=host,
        port=port,
        workers=1,
        log_level="info",
    )