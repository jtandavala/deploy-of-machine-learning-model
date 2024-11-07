import numpy as np
from config.core import config
from pipeline import price_pipe
from processing.data_manager import load_dataset, save_pipeline
from sklearn.model_selection import train_test_split


def run_training() -> None:
    """Train the model"""

    # read training data
    data = load_dataset(filename=config.app_config.training_data_file)

    # Convert categorical variables to the correct type
    for var in config.model_settings.categorical_vars:
        data[var] = data[var].astype("object")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_settings.features],
        data[config.model_settings.target],
        test_size=config.model_settings.test_size,
        random_state=config.model_settings.random_state,
    )

    # Log transform the target
    y_train = np.log(y_train)

    price_pipe.fit(X_train, y_train)
    save_pipeline(pipeline_to_persist=price_pipe)


if __name__ == "__main__":
    run_training()
    print("Training completed successfully!")
