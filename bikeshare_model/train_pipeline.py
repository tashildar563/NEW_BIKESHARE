import mlflow
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from bikeshare_model.config.core import config
from bikeshare_model.pipeline import bikeshare_pipe
from bikeshare_model.processing.data_manager import load_dataset, save_pipeline

def run_training()-> None:
    
    data = load_dataset(file_name = config.app_config.training_data_file)
    
    x_train, x_test, y_train, y_test = train_test_split(
        data[config.model_config.features],
        data[config.model_config.target],
        test_size = config.model_config.test_size,
        random_state=config.model_config.random_state   
    )
    
    bikeshare_pipe.fit(x_train,y_train)
    
    y_pred = bikeshare_pipe.predict(x_test)
    print(f"Mean Squared Error: {mean_squared_error(y_test,y_pred)}")
    
    print(f"R2 Score: {r2_score(y_test,y_pred)}")
    
    save_pipeline(pipeline_to_persist = bikeshare_pipe)
    
    
if __name__ == "__main__":
    run_training()
    