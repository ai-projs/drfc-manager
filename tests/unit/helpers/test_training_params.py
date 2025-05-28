import os
import yaml
from drfc_manager.helpers.training_params import (
    _setting_envs,
    writing_on_temp_training_yml,
)


def test_setting_envs_metrics_prefix(monkeypatch):
    # Ensure default behavior without prefix
    monkeypatch.delenv("DR_LOCAL_S3_METRICS_PREFIX", raising=False)
    config = _setting_envs("20220101010101", "model")
    assert config["METRICS_S3_OBJECT_KEY"].endswith(
        "TrainingMetrics-20220101010101.json"
    )

    # With custom prefix
    monkeypatch.setenv("DR_LOCAL_S3_METRICS_PREFIX", "myprefix")
    config2 = _setting_envs("20220101010101", "model")
    assert config2["METRICS_S3_OBJECT_KEY"] == "myprefix/TrainingMetrics.json"


def test_writing_on_temp_training_yml(tmp_path, monkeypatch):
    # Redirect temp dir by setting HOME
    monkeypatch.setenv("HOME", str(tmp_path))
    result = writing_on_temp_training_yml("modelname")
    assert isinstance(result, list) and len(result) == 2
    yaml_key, local_yaml_path = result
    assert yaml_key.startswith("modelname/")
    # Check file was created
    assert os.path.exists(local_yaml_path)
    # Validate YAML contents
    with open(local_yaml_path) as f:
        data = yaml.safe_load(f)
    assert data.get("JOB_TYPE") == "TRAINING"
    # Cleanup
    os.remove(local_yaml_path)
