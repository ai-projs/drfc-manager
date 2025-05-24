class BaseExceptionTransformers(Exception):

  def __init__(self, msg: str = None, exception: Exception = None):
    self.exception = exception
    self.msg = msg

  def __str__(self):
    if self.exception:
      return (
        f"\n🚨 Transformer Error 🚨\n"
        f"Message: {self.msg}\n"
        f"Caused by: {repr(self.exception)}"
      )
    return f"\n🚨 Transformer Error 🚨\nMessage: {self.msg}"