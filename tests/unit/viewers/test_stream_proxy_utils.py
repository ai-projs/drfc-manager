import pytest
from drfc_manager.viewers.stream_proxy_utils import (
    parse_containers,
    get_target_config,
    build_stream_url,
    parse_content_type,
    format_error_text,
    build_health_response,
)
from drfc_manager.viewers.exceptions import StreamResponseError
from drfc_manager.utils.logging_config import get_logger


def test_parse_containers_valid(caplog):
    """Test parsing valid container configuration."""
    containers_str = '["container1", "container2", "container3"]'
    logger = get_logger(__name__)
    caplog.set_level("DEBUG")
    result = parse_containers(containers_str, logger)
    assert result == ["container1", "container2", "container3"]
    assert "containers_parsed" in caplog.text
    assert "count=3" in caplog.text


def test_parse_containers_invalid_json(caplog):
    """Test parsing invalid JSON."""
    containers_str = "invalid json"
    logger = get_logger(__name__)

    result = parse_containers(containers_str, logger)
    assert result == []
    assert "invalid_json" in caplog.text
    assert "containers_str='invalid json'" in caplog.text


def test_parse_containers_not_list(caplog):
    """Test parsing non-list container configuration."""
    containers_str = '{"not": "a list"}'
    logger = get_logger(__name__)

    result = parse_containers(containers_str, logger)
    assert result == []
    assert "invalid_container_config" in caplog.text
    assert 'containers_str=\'{"not": "a list"}\'' in caplog.text


def test_parse_containers_non_string_items(caplog):
    """Test parsing container configuration with non-string items."""
    containers_str = "[1, 2, 3]"
    logger = get_logger(__name__)

    result = parse_containers(containers_str, logger)
    assert result == []
    assert "invalid_container_config" in caplog.text
    assert "containers_str='[1, 2, 3]'" in caplog.text


def test_get_target_config_defaults():
    """Test getting target configuration with defaults."""
    host, port = get_target_config()
    assert host == "localhost"
    assert port == 9080  # Default port in implementation


def test_get_target_config_env(monkeypatch):
    """Test getting target configuration from environment."""
    monkeypatch.setenv("DR_TARGET_HOST", "test-host")
    monkeypatch.setenv("DR_TARGET_PORT", "9090")

    host, port = get_target_config()
    assert host == "test-host"
    assert port == 9090


def test_get_target_config_override():
    """Test getting target configuration with overrides."""
    host, port = get_target_config(host="override-host", port=7070)
    assert host == "override-host"
    assert port == 7070


def test_build_stream_url_defaults():
    """Test building stream URL with defaults."""
    url = build_stream_url(
        host="localhost",
        port=8080,
        topic="/racecar/deepracer/kvs_stream",
        quality=75,
        width=480,
        height=360,
    )
    assert (
        url
        == "http://localhost:8080/stream?topic=/racecar/deepracer/kvs_stream&quality=75&width=480&height=360"
    )


def test_build_stream_url_env(monkeypatch):
    """Test building stream URL with environment variables."""
    monkeypatch.setenv("DR_TARGET_HOST", "myhost")
    monkeypatch.setenv("DR_TARGET_PORT", "1234")
    url = build_stream_url(
        host="myhost",
        port=1234,
        topic="/racecar/deepracer/kvs_stream",
        quality=80,
        width=640,
        height=480,
    )
    assert (
        url
        == "http://myhost:1234/stream?topic=/racecar/deepracer/kvs_stream&quality=80&width=640&height=480"
    )


def test_build_stream_url_override():
    """Test building stream URL with overrides."""
    url = build_stream_url(
        topic="test-container",
        quality=75,
        width=640,
        height=480,
        host="override-host",
        port=7070,
    )
    assert (
        url
        == "http://override-host:7070/stream?topic=test-container&quality=75&width=640&height=480"
    )


def test_parse_content_type_str():
    """Test parsing string content type."""
    content_type = "video/mp4"
    result = parse_content_type(content_type)
    assert result == ("video/mp4", "video/mp4")


def test_parse_content_type_bytes():
    """Test parsing bytes content type."""
    content_type = b"video/mp4"
    result = parse_content_type(content_type)
    assert result == ("video/mp4", "video/mp4")


def test_parse_content_type_none():
    """Test parsing None content type."""
    result = parse_content_type(None)
    assert result == ("image/jpeg", "image/jpeg")


def test_parse_content_type_invalid():
    """Test parsing invalid content type."""
    with pytest.raises(StreamResponseError):
        parse_content_type(123)


def test_format_error_text_short():
    """Test formatting short error text."""
    text = b"Short error message"
    result = format_error_text(text)
    assert result == "Short error message"


def test_format_error_text_truncate():
    """Test truncating long error text."""
    data = b"x" * 200
    text = format_error_text(data, max_length=100)
    assert len(text) <= 103  # Allow for ellipsis and actual implementation
    assert text.endswith("...")


def test_format_error_text_invalid_utf8():
    """Test handling invalid UTF-8."""
    data = b"\xff\xfe"
    result = format_error_text(data)
    assert result == ""


def test_build_health_response_healthy():
    """Test building healthy response."""
    response = build_health_response(
        target_host="test-host",
        target_port=8080,
        socket_status="open",
        ping_status="ok",
        containers=["container1"],
        error_details={},
        target_reachable=True,
        target_responsive=True,
    )
    details = response["details"]
    assert response["status"] == "healthy"
    assert details["target_stream_server"]["host"] == "test-host"
    assert details["target_stream_server"]["port"] == 8080
    assert details["known_containers_count"] == 1
    assert "errors" not in response


def test_build_health_response_unhealthy():
    """Test building unhealthy response."""
    response = build_health_response(
        target_host="test-host",
        target_port=8080,
        socket_status="closed",
        ping_status="error",
        containers=[],
        error_details={"socket": "Connection refused"},
        target_reachable=False,
        target_responsive=False,
    )
    details = response["details"]
    assert response["status"] == "unhealthy"
    assert details["target_stream_server"]["host"] == "test-host"
    assert details["target_stream_server"]["port"] == 8080
    assert details["known_containers_count"] == 0
    assert "errors" in response
    assert response["errors"]["socket"] == "Connection refused"
