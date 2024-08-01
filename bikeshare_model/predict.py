import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent,file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np

from bikeshare_model import __version__ as __version
from bikeshare_model.config.core import config
from bikeshare_model.processing.data_manager import load_pipeline
from bikeshare_model.processing.validation import validate_inputs
from bikeshare_model.processing.validation import pre_pipeline_preparation

pipeline_file_name = f"{config.app_config.pipeline_save_file}{__version}.pkl"
bikeshare_pipe = load_pipeline(file_name = pipeline_file_name)

def make_prediction(*, input_data: Union[pd.DataFrame,dict])-> dict:
    
    validated_Data, errors = validate_inputs(input_df = pd.DataFrame(input_data))
    validated_Data = validated_Data.reindex(columns = config.model_config.features)
    
    results = {"predictions":None,"version":__version, "errors":errors}
    
    if not errors:
        predictions = bikeshare_pipe.predict(validated_Data)
        results = {"predictions":predictions} 
        print(results)
    return results
if __name__ == "__main__":
    data_in = {'dteday': ['2012-11-6'], 'season': ['winter'], 'hr': ['6pm'], 'holiday': ['No'], 'weekday': ['Tue'],
               'workingday': ['Yes'], 'weathersit': ['Clear'], 'temp': [16], 'atemp': [17.5], 'hum': [30], 'windspeed': [10]}
    print(make_prediction(input_data = data_in))