import pytest
from drfc_manager.config_env import MinioConfig


@pytest.mark.parametrize(
    "input_url,expected",
    [
        ("minio:9000", "http://minio:9000"),
        ("http://example.com", "http://example.com"),
        ("https://example.com", "https://example.com"),
    ],
)
def test_ensure_http_scheme(input_url, expected):
    conf = MinioConfig(server_url=input_url)
    assert conf.server_url == expected


def test_default_server_url_has_http():
    conf = MinioConfig()
    assert conf.server_url.startswith("http://") or conf.server_url.startswith(
        "https://"
    )
