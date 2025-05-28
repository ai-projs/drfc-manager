import os
import pytest
from drfc_manager.utils.paths import (
    ensure_dir_exists,
    get_internal_path,
    INTERNAL_DIRS,
    get_comms_dir,
    get_logs_dir,
    get_docker_compose_path,
)


def test_ensure_dir_exists(tmp_path):
    dir_path = tmp_path / "testdir"
    ensure_dir_exists(dir_path)
    assert dir_path.exists()
    # perms are set to 0o777
    mode = os.stat(dir_path).st_mode & 0o777
    assert mode == 0o777


def test_get_internal_path_unknown():
    with pytest.raises(ValueError):
        get_internal_path("INVALID")


def test_get_internal_path_creates_and_returns(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "drfc_manager.utils.paths.PACKAGE_ROOT", tmp_path, raising=False
    )
    # override INTERNAL_DIRS entry to point to tmp_path
    monkeypatch.setitem(INTERNAL_DIRS, "testdir", tmp_path)
    path = get_internal_path("testdir", "sub1", "sub2")
    expected = tmp_path / "sub1" / "sub2"
    assert path == expected
    assert path.exists()


def test_get_comms_dir_creates_and_returns(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "drfc_manager.utils.paths.PACKAGE_ROOT", tmp_path, raising=False
    )
    # Override the INTERNAL_DIRS entry for 'comms'
    monkeypatch.setitem(INTERNAL_DIRS, "comms", tmp_path)
    path = get_comms_dir(42)
    expected = tmp_path / "42"
    assert path == expected
    assert path.exists()


def test_get_logs_dir_creates_and_returns(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "drfc_manager.utils.paths.PACKAGE_ROOT", tmp_path, raising=False
    )
    # Override the INTERNAL_DIRS entry for 'logs'
    monkeypatch.setitem(INTERNAL_DIRS, "logs", tmp_path)
    path = get_logs_dir("modelX")
    expected = tmp_path / "robomaker" / "modelX"
    assert path == expected
    assert path.exists()


def test_get_docker_compose_path_success(tmp_path, monkeypatch):
    # Override the INTERNAL_DIRS entry for 'docker_composes'
    monkeypatch.setitem(INTERNAL_DIRS, "docker_composes", tmp_path)
    # Create a dummy compose file
    compose_file = tmp_path / "docker-compose-test.yml"
    compose_file.write_text("")
    path = get_docker_compose_path("test")
    assert path == compose_file


def test_get_docker_compose_path_not_found(tmp_path, monkeypatch):
    # Override the INTERNAL_DIRS entry for 'docker_composes'
    monkeypatch.setitem(INTERNAL_DIRS, "docker_composes", tmp_path)
    with pytest.raises(FileNotFoundError):
        get_docker_compose_path("missing")
