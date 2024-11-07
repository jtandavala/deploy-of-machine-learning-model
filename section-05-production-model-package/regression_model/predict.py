import typing as t

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from regression_model import __version__ as _version
from regression_model.config.core import config
from regression_model.processing.data_manager import load_pipeline, load_dataset
from regression_model.processing.validation import validate_inputs

pipeline_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_price_pipe = load_pipeline(filename=pipeline_file_name)


def make_predictions(*, input_data: t.Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model pipeline"""

    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)

    results = {
        "predictions": [],
        "version": _version,
        "errors": errors,
    }

    if not errors:
        predictions = _price_pipe.predict(
            X=validated_data[config.model_settings.features]
        )
        results["predictions"] = [np.exp(pred) for pred in predictions]

    return results


input_data = {
    "MSSubClass": [60],
    "MSZoning": ["RL"],
    "LotFrontage": [65],
    "LotShape": ["Reg"],
    "LandContour": ["Lvl"],
    "LotConfig": ["Inside"],
    "Neighborhood": ["CollgCr"],
    "OverallQual": [7],
    "OverallCond": [5],
    "YearRemodAdd": [2003],
    "RoofStyle": ["Gable"],
    "Exterior1st": ["VinylSd"],
    "ExterQual": ["Gd"],
    "Foundation": ["PConc"],
    "BsmtQual": ["Gd"],
    "BsmtExposure": ["No"],
    "BsmtFinType1": ["GLQ"],
    "HeatingQC": ["Ex"],
    "CentralAir": ["Y"],
    "FirstFlrSF": [856],  # Added FirstFlrSF
    "SecondFlrSF": [854],  # Added SecondFlrSF
    "GrLivArea": [1710],
    "BsmtFullBath": [1],
    "HalfBath": [1],
    "KitchenQual": ["Gd"],
    "TotRmsAbvGrd": [8],
    "Functional": ["Typ"],
    "Fireplaces": [1],
    "FireplaceQu": ["Gd"],
    "GarageFinish": ["RFn"],
    "GarageCars": [2],
    "GarageArea": [548],
    "PavedDrive": ["Y"],
    "WoodDeckSF": [0],
    "ScreenPorch": [0],
    "SaleCondition": ["Normal"],
    "YrSold": [2008],  # Temporal variable
}


results = make_predictions(input_data=input_data)
print(results)
