import os
import sys
import logging
import tempfile
from datetime import datetime
from functools import wraps
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger('drfc')
logger.propagate = False

LOG_DIR = os.path.join(tempfile.gettempdir(), "drfc_logs")
os.makedirs(LOG_DIR, exist_ok=True)

def setup_logging(run_id: Optional[int] = None, model_name: Optional[str] = None, quiet: bool = True):
    """
    Setup logging configuration with file output.
    
    Args:
        run_id: Run ID to include in log filename
        model_name: Model name to include in log filename
        quiet: If True, only warnings and errors go to console (default: True)
    """
    logger.handlers.clear()    

    logger.addHandler(logging.NullHandler())
    
    if not quiet:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')
        console.setFormatter(console_formatter)
        logger.addHandler(console)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    components = ["drfc"]
    if model_name:
        components.append(model_name)
    if run_id is not None:
        components.append(f"run{run_id}")
    components.append(timestamp)
    
    log_filename = "_".join(components) + ".log"
    log_path = os.path.join(LOG_DIR, log_filename)
    
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    logger.debug(f"Logging to file: {log_path}")
    return log_path

logger.addHandler(logging.NullHandler())

def get_recent_logs(n: int = 5):
    """Get paths to the n most recent log files."""
    if not os.path.exists(LOG_DIR):
        return []
    
    log_files = [os.path.join(LOG_DIR, f) for f in os.listdir(LOG_DIR) 
                if f.startswith('drfc_') and f.endswith('.log')]
    
    log_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    return log_files[:n]

def log_execution(func):
    """Decorator to log function execution with arguments and result."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"Executing {func.__name__} with args={args}, kwargs={kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.exception(f"Error in {func.__name__}: {e}")
            raise
    return wrapper 