import os
import time
import pytest
from drfc_manager.utils.logging import setup_logging, get_recent_logs, log_execution


def test_setup_logging_creates_log_file(tmp_path, monkeypatch):
    # Override LOG_DIR to isolated tmp path
    monkeypatch.setattr(
        "drfc_manager.utils.logging.LOG_DIR", str(tmp_path), raising=False
    )
    os.makedirs(str(tmp_path), exist_ok=True)
    log_path = setup_logging(run_id=99, model_name="testmodel", quiet=True)
    assert os.path.exists(log_path)
    assert log_path.endswith(".log")
    # Cleanup
    os.remove(log_path)


def test_get_recent_logs(tmp_path, monkeypatch):
    # Override LOG_DIR and create dummy log files
    monkeypatch.setattr(
        "drfc_manager.utils.logging.LOG_DIR", str(tmp_path), raising=False
    )
    os.makedirs(str(tmp_path), exist_ok=True)
    # Create drfc_ log files and one non-drfc file
    file1 = tmp_path / "drfc_file1.log"
    file1.write_text("log1")
    time.sleep(0.01)
    file2 = tmp_path / "drfc_file2.log"
    file2.write_text("log2")
    time.sleep(0.01)
    other = tmp_path / "other.log"
    other.write_text("ignore")
    recent = get_recent_logs(n=2)
    # Should only include drfc_ prefixed files
    assert len(recent) == 2
    assert all(os.path.basename(p).startswith("drfc_") for p in recent)
    assert set(recent) == {str(file2), str(file1)}


def test_log_execution_decorator_success():
    @log_execution
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_log_execution_decorator_exception():
    @log_execution
    def fail():
        raise ValueError("failure")

    with pytest.raises(ValueError):
        fail()
