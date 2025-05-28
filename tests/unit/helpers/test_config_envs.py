import os
import pytest
from drfc_manager.helpers.config_envs import (
    _discover_path_to_config_envs,
    load_envs_from_files,
    find_envs_files,
)
from drfc_manager.types.config import ConfigEnvs


def test_discover_path_to_config_envs(tmp_path, monkeypatch):
    # Build nested project structure with config directory
    project = tmp_path / "project"
    nested = project / "a" / "b"
    nested.mkdir(parents=True)
    config_dir = project / "config"
    config_dir.mkdir()

    monkeypatch.chdir(str(nested))
    path = _discover_path_to_config_envs()
    assert path.endswith(os.path.join("project", "config") + os.sep)


def test_load_envs_from_files(tmp_path):
    env_file = tmp_path / "test.env"
    env_file.write_text("FOO=BAR\n")
    load_envs_from_files([str(env_file)])
    assert os.environ.get("FOO") == "BAR"

    with pytest.raises(FileNotFoundError):
        load_envs_from_files([str(env_file) + ".doesnotexist"])


def test_find_envs_files(monkeypatch):
    # Stub discovery to return our fake config path
    fake_path = "/tmp/config/"
    monkeypatch.setattr(
        "drfc_manager.helpers.config_envs._discover_path_to_config_envs",
        lambda: fake_path,
    )

    files = find_envs_files([ConfigEnvs.run, ConfigEnvs.system])
    assert files == [fake_path + "run.env", fake_path + "system.env"]
