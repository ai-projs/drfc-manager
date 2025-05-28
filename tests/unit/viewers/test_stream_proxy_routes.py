"""Tests for stream proxy route handlers."""

import pytest
import httpx
from unittest.mock import patch, AsyncMock

from drfc_manager.viewers.stream_proxy_routes import (
    validate_container_id,
    check_socket_connection,
    check_http_ping,
)
from drfc_manager.viewers.exceptions import (
    UnknownContainerError,
    StreamProxySocketError,
    StreamProxyPingError,
)


def test_validate_container_id():
    """Test container ID validation."""
    # Test valid container ID
    containers = ["container1", "container2"]
    validate_container_id("container1", containers)  # Should not raise

    # Test invalid container ID
    with pytest.raises(UnknownContainerError):
        validate_container_id("unknown", containers)

    # Test empty containers list (should not raise)
    validate_container_id("any_id", [])  # Should not raise since containers is empty


@pytest.mark.asyncio
async def test_check_socket_connection():
    """Test socket connection checks."""
    # Test successful connection
    with patch("socket.create_connection") as mock_conn:
        mock_conn.return_value.__enter__.return_value = None
        result = await check_socket_connection("localhost", 8080)
        assert result[0] is True
        assert result[1] == "open"
        assert result[2] == {}

    # Test connection refused (when service is down)
    with patch("socket.create_connection") as mock_conn:
        mock_conn.side_effect = ConnectionRefusedError()
        with pytest.raises(StreamProxySocketError) as exc_info:
            await check_socket_connection("localhost", 8080)
        assert "Connection refused" in str(exc_info.value)


@pytest.mark.asyncio
async def test_check_http_ping():
    """Test HTTP ping checks."""
    mock_client = AsyncMock()

    # Test successful ping (normal operation)
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_client.get.return_value = mock_response
    result = await check_http_ping(mock_client, "http://localhost:8080")
    assert result[0] is True
    assert "ok" in result[1]
    assert result[2] == {}

    # Test service unavailable (503 - common in container orchestration)
    mock_response = AsyncMock()
    mock_response.status_code = 503
    mock_client.get.return_value = mock_response
    with pytest.raises(StreamProxyPingError) as exc_info:
        await check_http_ping(mock_client, "http://localhost:8080")
    assert "server error status" in str(exc_info.value)

    # Test connection refused (service not listening)
    mock_client.get.side_effect = httpx.ConnectError("Connection refused")
    with pytest.raises(StreamProxyPingError) as exc_info:
        await check_http_ping(mock_client, "http://localhost:8080")
    assert "HTTP Connection refused" in str(exc_info.value)
