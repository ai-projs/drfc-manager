# Server configuration
DEFAULT_PROXY_PORT = 8090
DEFAULT_TARGET_HOST = "localhost"
DEFAULT_TARGET_PORT = 8080

# Stream configuration
DEFAULT_QUALITY = 75
DEFAULT_WIDTH = 480
DEFAULT_HEIGHT = 360
DEFAULT_TOPIC = "/racecar/deepracer/kvs_stream"

# HTTP client configuration
HTTPX_TIMEOUT_CONNECT = 10.0
HTTPX_TIMEOUT_READ = 30.0
HTTPX_STREAM_CHUNK_SIZE = 65536

# Health check configuration
HEALTH_CHECK_SOCKET_TIMEOUT = 2.0
HEALTH_CHECK_PING_TIMEOUT = 5.0

# Stream proxy utils configuration
DEFAULT_CONTENT_TYPE = "image/jpeg"
DEFAULT_MEDIA_TYPE = "image/jpeg"
