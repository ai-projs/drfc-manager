import os
import uvicorn
from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware

from drfc_manager.viewers.stream_proxy_routes import proxy_stream, health_check
from drfc_manager.viewers.stream_proxy_utils import parse_containers
from drfc_manager.utils.logging_config import get_logger
from drfc_manager.types.constants import (
    DEFAULT_PROXY_PORT,
    DEFAULT_TOPIC,
    DEFAULT_QUALITY,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
)

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="DeepRacer Stream Proxy")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    containers = parse_containers(os.environ.get("DR_VIEWER_CONTAINERS", "[]"))

    @app.get("/{container_id}/stream")
    async def stream_route(
        request: Request,
        container_id: str,
        topic: str = Query(DEFAULT_TOPIC, description="ROS topic to stream"),
        quality: int = Query(
            DEFAULT_QUALITY, description="Image quality (1-100)", ge=1, le=100
        ),
        width: int = Query(DEFAULT_WIDTH, description="Image width", ge=1),
        height: int = Query(DEFAULT_HEIGHT, description="Image height", ge=1),
    ):
        return await proxy_stream(
            request,
            container_id,
            containers,
            topic=topic,
            quality=quality,
            width=width,
            height=height,
        )

    @app.get("/health")
    async def health_route():
        return await health_check(containers)

    return app


def main():
    """Main entry point for the stream proxy server."""
    port = int(os.environ.get("DR_PROXY_PORT", DEFAULT_PROXY_PORT))
    host = "0.0.0.0"

    logger.info("starting_proxy_server", host=host, port=port)

    containers = parse_containers(os.environ.get("DR_VIEWER_CONTAINERS", "[]"))
    if containers:
        logger.info(
            "container_config_loaded",
            container_count=len(containers),
            containers=containers,
        )
    else:
        logger.info(
            "no_container_config",
            message="No specific container IDs loaded. Proxying requests for any container ID.",
        )

    app = create_app()
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=1,
        log_level="info",
    )


app = create_app()

if __name__ == "__main__":
    main()
