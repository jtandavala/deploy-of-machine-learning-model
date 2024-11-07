from feature_engine.encoding import OrdinalEncoder, RareLabelEncoder
from feature_engine.imputation import (
    AddMissingIndicator,
    CategoricalImputer,
    MeanMedianImputer,
)
from feature_engine.selection import DropFeatures
from feature_engine.transformation import LogTransformer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer

from regression_model.config.core import config
from regression_model.processing import features as pp
from regression_model.processing.data_manager import convert_categorical_to_object


price_pipe = Pipeline(
    [
        # impute categorical variables with string 'missing'
        (
            "missing_imputation",
            CategoricalImputer(
                imputation_method="missing",
                variables=config.model_settings.categorical_vars_with_na_missing,
            ),
        ),
        (
            "frequent_imputation",
            CategoricalImputer(
                imputation_method="frequent",
                variables=config.model_settings.categorical_vars_with_na_frequent,
            ),
        ),
        # Add missing indicator
        (
            "missing_indicator",
            AddMissingIndicator(variables=config.model_settings.numerical_vars_with_na),
        ),
        # impute numerical variables with the mean
        (
            "mean_imputation",
            MeanMedianImputer(
                imputation_method="mean",
                variables=config.model_settings.numerical_vars_with_na,
            ),
        ),
        # == TEMPORAL VARIABLES ==
        (
            "elapsed_time",
            pp.TemporalVariableTransformer(
                variables=config.model_settings.temporal_vars,
                reference_variable=config.model_settings.ref_var,
            ),
        ),
        (
            "drop_features",
            DropFeatures(features_to_drop=[config.model_settings.ref_var]),
        ),
        # == VARIABLE TRANSFORMATION ==
        ("log", LogTransformer(variables=config.model_settings.numericals_log_vars)),
        (
            "binarizer",
            SklearnTransformerWrapper(
                transformer=Binarizer(threshold=0),
                variables=config.model_settings.binarize_vars,
            ),
        ),
        # == mappers ==
        (
            "mapper_qual",
            pp.Mapper(
                variables=config.model_settings.qual_vars,
                mappings=config.model_settings.qual_mappings,
            ),
        ),
        (
            "mapper_exposure",
            pp.Mapper(
                variables=config.model_settings.exposure_vars,
                mappings=config.model_settings.exposure_mappings,
            ),
        ),
        (
            "mapper_finish",
            pp.Mapper(
                variables=config.model_settings.finish_vars,
                mappings=config.model_settings.finish_mappings,
            ),
        ),
        (
            "mapper_garage",
            pp.Mapper(
                variables=config.model_settings.garage_vars,
                mappings=config.model_settings.garage_mappings,
            ),
        ),
        # == CATEGORICAL ENCODING ==
        (
            "type_converter",
            FunctionTransformer(convert_categorical_to_object, validate=False),
        ),
        (
            "rare_label_encoder",
            RareLabelEncoder(
                tol=0.01,
                n_categories=1,
                variables=config.model_settings.categorical_vars,
            ),
        ),
        # encode categorical using the target mean
        (
            "categorical_encoder",
            OrdinalEncoder(
                encoding_method="ordered",
                variables=config.model_settings.categorical_vars,
            ),
        ),
        (
            "scaler",
            pp.CustomMixMaxScaler(),
        ),
        (
            "final_imputer",
            SklearnTransformerWrapper(
                transformer=SimpleImputer(strategy="constant", fill_value=0),
            ),
        ),
        (
            "lasso",
            Lasso(
                alpha=config.model_settings.alpha,
                random_state=config.model_settings.random_state,
            ),
        ),
    ]
)
