import os

from minio import Minio as MinioClient


access_key = os.getenv('ACCESS_KEY')
secret_key = os.getenv('SECRET_KEY')
endpoint = os.getenv('MINIO_SERVER_URL')


class MinioClientServer:
    _instance = None
    
    def __init__(self):
        raise RuntimeError("This is a Singleton class, invoke the get_instance() method instead")
    
    @classmethod
    def get_instance(cls):
        try:
            if cls._instance is None:
                cls._instance = MinioClient(endpoint, access_key, secret_key, secure=False)
            return cls._instance
        except Exception as e:
            raise e
