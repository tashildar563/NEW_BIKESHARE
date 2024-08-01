import sys
from pathlib import Path
file =Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import typing as t
import re
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from bikeshare_model import __version__ as _version
from bikeshare_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

## pre-pipeline preparation 
# extract year and month from the date column and create two another columns

def get_year_and_month(dataframe: pd.DataFrame, date_var:str):
    df = dataframe.copy()
    
    df[date_var] = pd.to_datetime(df[date_var], format='%Y-%m-%d')
    
    #adding new feature yr and mnth
    df['yr'] = df[date_var].dt.year
    df['mnth']=df[date_var].dt.month_name()
    
    return df

def pre_pipeline_preparation(*,data_frame:pd.DataFrame)-> pd.DataFrame:
    data_frame =  get_year_and_month(dataframe=data_frame, date_var=config.model_config.date_var)
    
    for field in config.model_config.unused_fields:
        if field in data_frame.columns:
            data_frame.drop(labels = field, axis=1,inplace=True)

    return data_frame

def _load_raw_dataset(*,file_name:str)->pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe

def load_dataset(*,file_name:str)-> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = pre_pipeline_preparation(data_frame=dataframe) 
    return transformed

def save_pipeline(*,pipeline_to_persist:Pipeline)->None:
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR/save_file_name
    
    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist,save_path) 
    
def load_pipeline(*,file_name:str)->Pipeline:
    file_path = TRAINED_MODEL_DIR/file_name
    trained_model = joblib.load(file_path)
    return trained_model

def remove_old_pipelines(*,files_to_keep:t.List[str])->None:
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()
              