import urllib.parse
from urllib.parse import urljoin

def minio_console_link(minio_url: str, bucket_name: str, object_path: str) -> str:
    minio_console_url = minio_url.replace("9000", "9001").rstrip('/')
    object_path_encoded = urllib.parse.quote(object_path, safe='')
    return urljoin(minio_console_url + '/', f"browser/{bucket_name}/{object_path_encoded}")