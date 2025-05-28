import pytest
from unittest.mock import AsyncMock
import httpx


class AsyncIterator:
    def __init__(self, items):
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


class ErrorAsyncIterator:
    def __init__(self, error):
        self.error = error

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise self.error


@pytest.fixture
def normal_response():
    mock = AsyncMock()
    mock.aiter_bytes = lambda chunk_size=None: AsyncIterator([b"chunk1", b"chunk2"])
    return mock


@pytest.fixture
def empty_response():
    mock = AsyncMock()
    mock.aiter_bytes = lambda chunk_size=None: AsyncIterator([])
    return mock


@pytest.fixture
def read_error_response():
    mock = AsyncMock()
    mock.aiter_bytes = lambda chunk_size=None: ErrorAsyncIterator(
        httpx.ReadError("Test error")
    )
    return mock


@pytest.fixture
def unexpected_error_response():
    mock = AsyncMock()
    mock.aiter_bytes = lambda chunk_size=None: ErrorAsyncIterator(
        Exception("Unexpected error")
    )
    return mock
