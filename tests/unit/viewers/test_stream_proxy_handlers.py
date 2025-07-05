import pytest
from drfc_manager.viewers.stream_proxy_handlers import create_stream_generator


@pytest.mark.asyncio
async def test_create_stream_generator(
    normal_response, empty_response, read_error_response, unexpected_error_response
):
    # Normal case
    chunks = []
    async for chunk in create_stream_generator(normal_response, "test-container"):
        chunks.append(chunk)
    assert chunks == [b"chunk1", b"chunk2"]

    # Empty response
    chunks = []
    async for chunk in create_stream_generator(empty_response, "test-container"):
        chunks.append(chunk)
    assert chunks == []

    # Read error
    chunks = []
    async for chunk in create_stream_generator(read_error_response, "test-container"):
        chunks.append(chunk)
    assert chunks == []

    # Unexpected error
    chunks = []
    async for chunk in create_stream_generator(
        unexpected_error_response, "test-container"
    ):
        chunks.append(chunk)
    assert chunks == []
