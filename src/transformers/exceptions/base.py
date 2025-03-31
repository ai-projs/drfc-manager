class BaseExceptionTransformers(Exception):
    
    def __init__(self, msg: str = None, exception: Exception = None):
        self.exception = exception
        self.msg = msg
        
    def __str__(self):
        return f'{self.msg}, {self.exception}'
        
    
