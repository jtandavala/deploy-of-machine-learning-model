import numpy as np
from config.core import config
from pipeline import price_pipe
from processing.data_manager import load_dataset, save_pipeline
from sklearn.model_selection import train_test_split


def run_training() -> None:
    """Train the model"""
    
    # read training data
    data = load_dataset(filename=config.app_config.training_data_file)
    print(data)
    


if __name__ == "__main__":
    run_training()



