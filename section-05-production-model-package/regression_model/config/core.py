from pathlib import Path
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel, ConfigDict
from strictyaml import YAML, load

import regression_model

# Project Directories
PACKAGE_ROOT = Path(regression_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    """
    Application-level config
    """

    model_config = ConfigDict(strict=True, frozen=True)

    package_name: str
    training_data_file: str
    test_data_file: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        protected_namespaces=(),  # This removes the warning about model_ prefix
    )

    target: str
    variables_to_rename: Dict
    features: List[str]
    test_size: float
    random_state: int
    alpha: float
    categorical_vars_with_na_frequent: List[str]
    categorical_vars_with_na_missing: List[str]
    numerical_vars_with_na: List[str]
    temporal_vars: List[str]
    ref_var: str
    numericals_log_vars: Sequence[str]
    binarize_vars: Sequence[str]
    qual_vars: List[str]
    exposure_vars: List[str]
    finish_vars: List[str]
    garage_vars: List[str]
    categorical_vars: List[str]
    qual_mappings: Dict[str, int]
    exposure_mappings: Dict[str, int]
    garage_mappings: Dict[str, int]
    finish_mappings: Dict[str, int]


class MasterConfig(BaseModel):
    """Master config object"""

    model_config = ConfigDict(
        strict=True,
        frozen=True,
        protected_namespaces=(),  # This removes the warning about model_ prefix
    )

    app_config: AppConfig
    model_settings: ModelConfig


def find_config_file() -> Path:
    """Locate the configuration file"""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH}")


def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:
    """Parse YAML containing the package configuration."""
    if not cfg_path:
        cfg_path = find_config_file()
    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> MasterConfig:
    """Run validation on config values"""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    # Convert string values to appropriate types
    config_data = parsed_config.data

    # Convert numeric strings to actual numbers
    config_data["test_size"] = float(config_data["test_size"])
    config_data["random_state"] = int(config_data["random_state"])
    config_data["alpha"] = float(config_data["alpha"])

    # Convert mapping values to integers
    for mapping_key in [
        "qual_mappings",
        "exposure_mappings",
        "garage_mappings",
        "finish_mappings",
    ]:
        if mapping_key in config_data:
            config_data[mapping_key] = {
                k: int(v) for k, v in config_data[mapping_key].items()
            }

    # Create the config using model_validate
    _config = MasterConfig(
        app_config=AppConfig.model_validate(config_data),
        model_settings=ModelConfig.model_validate(config_data),
    )
    return _config


config = create_and_validate_config()
