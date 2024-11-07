import typing as t
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


from regression_model import __version__ as _version
from regression_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

pd.set_option("future.no_silent_downcasting", True)


def load_dataset(*, filename: str) -> pd.DataFrame:
    df = pd.read_csv(Path(f"{DATASET_DIR}/{filename}"))
    df["MSSubClass"] = df["MSSubClass"].astype("O")

    # rename variables beginning with numbers to avoid
    transformed = df.rename(columns=config.model_settings.variables_to_rename)
    return transformed


def convert_categorical_to_object(X):
    """Convert specified columns to object dtype"""
    X = pd.DataFrame(X, columns=config.model_settings.features)
    for var in config.model_settings.categorical_vars:
        X[var] = X[var].astype("object")
    return X


def remove_old_pipeline(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipeline
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipeline(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)
