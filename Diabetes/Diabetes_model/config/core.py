import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Dict, List
from pydantic import BaseModel
from strictyaml import YAML, load

import Diabetes_model

# Project Directories
PACKAGE_ROOT = Path(Diabetes_model.__file__).resolve().parent
#print(PACKAGE_ROOT)
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
#print(CONFIG_FILE_PATH)

DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"

class Mcfg(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    features: List[str]
    target: str

    test_size : float
    colsample_bytree: float
    learning_rate: float
    max_depth: int
    n_estimators: int
    scale_pos_weight: int
    subsample: float


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    training_data_file: str
    test_data_file: str
    pipeline_save_file: str


class Config(BaseModel):
    """Master config object."""
    app_config: AppConfig
    models_cfg: Mcfg


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    raise Exception(f"Config not found at {CONFIG_FILE_PATH!r}")


def fetch_config_from_yaml(cfg_path: Path = None) -> YAML:
    """Parse YAML containing the package configuration."""

    if not cfg_path:
        cfg_path = find_config_file()

    if cfg_path:
        with open(cfg_path, "r") as conf_file:
            parsed_config = load(conf_file.read())
            return parsed_config
    raise OSError(f"Did not find config file at path: {cfg_path}")


def create_and_validate_config(parsed_config: YAML = None) -> Config:
    """Run validation on config values."""
    if parsed_config is None:
        parsed_config = fetch_config_from_yaml()

    
    # specify the data attribute from the strictyaml YAML type.
    try:
        _config = Config(
            app_config=AppConfig(**parsed_config.data),
            models_cfg=Mcfg(**parsed_config.data),
        )
    except ValueError as e:
        print(e.errors())

    return _config


config = create_and_validate_config()
