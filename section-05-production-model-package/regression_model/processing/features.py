from typing import List

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import Binarizer, MinMaxScaler, FunctionTransformer


class TemporalVariableTransformer(BaseEstimator, TransformerMixin):
    """Temporal elapsed time transformer"""

    def __init__(self, variables: List[str], reference_variable: str):
        if not isinstance(variables, list):
            raise ValueError("variables should a list")
        self.variables = variables
        self.reference_variable = reference_variable

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # we need this step to fit the sklearn pipeline
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # so that we do not over-write the original dataframe
        X = X.copy()

        for feature in self.variables:
            X[feature] = X[self.reference_variable] - X[feature]
        return X


class Mapper(BaseEstimator, TransformerMixin):
    """Categorical variable mapper."""

    def __init__(self, variables: List[str], mappings: dict):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        self.variables = variables
        self.mappings = mappings

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[feature].replace(self.mappings).astype(X[feature].dtype)
        return X


class DataPrinter(BaseEstimator, TransformerMixin):
    """Custom transformer that prints data during pipeline execution"""

    def __init__(self, message="", features=None):
        self.message = message
        self.features = features

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        print(f"\n--- DataPrinter Fit: {self.message} ---")
        if self.features:
            print(X[self.features].head())
        else:
            print(X.head())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        print(f"\n--- DataPrinter Transform: {self.message} ---")
        if self.features:
            print(X[self.features].head())
        else:
            print(X.head())
        return X


class CustomMixMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.feature_names = None

    def fit(self, X, y=None):
        # Store feature names if X is a DataFrame
        self.feature_names = X.columns if isinstance(X, pd.DataFrame) else None

        # Create a copy and fill NaN with 0 temporarily for fitting
        X_copy = pd.DataFrame(X).copy()
        X_copy = X_copy.fillna(0)

        self.scaler.fit(X_copy)
        return self

    def transform(self, X):
        # Create a copy and fill NaN with 0 temporarily for transformation
        X_copy = pd.DataFrame(X).copy()
        X_copy = X_copy.fillna(0)

        # Transform the data
        X_scaled = self.scaler.transform(X_copy)

        # Convert back to DataFrame if input was DataFrame
        if self.feature_names is not None:
            X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)

        return X_scaled
