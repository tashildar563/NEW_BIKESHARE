from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from typing import List
import sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class WeekdayImputer(BaseEstimator,TransformerMixin):
    
    #impute missing  values in weekdays column by extracting dayname from dteday column#
    
    def __init__(self,variable:str,date_var:str):
        if not isinstance(variable,str):
            raise ValueError("variable should be a string")
        if not isinstance(date_var,str):
            raise ValesError("date_var s name should be a String")
        
        self.variable = variable 
        self.date_var = date_var
        
    def fit(self, X:pd.DataFrame, y:pd.Series=None):
        return self
    
    def transform(self, X:pd.DataFrame)-> pd.DataFrame:
        x= X.copy()
        
        x[self.date_var]=pd.to_datetime(x[self.date_var],format='%y-%m-%d')
        
        wday_null_idx = x[x[self.variable].isnull() == True].index
        x.loc[wday_null_idx,self.variable] = x.loc[wday_null_idx,self.date_var].dt.day_name().apply(lambda x:x[:3])
        
        #drop dteday column after imputation#
        
        x.drop(self.date_var,axis=1,inplace=True)
        
        return x
        
        
        

class WeathersitImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self,variable:str):
        if not isinstance(variable,str):
            raise ValueError("Variable should be a string")
        self.variable=variable
    
    def fit(self,X:pd.DataFrame,y:pd.Series=None):
        x=X.copy()
        self.fill_value = x[self.variable].mode()[0]
        
        return self
    
    def transform(self,X:pd.DataFrame)-> pd.DataFrame:
        x=X.copy()
        x[self.variable] = x[self.variable].fillna(self.fill_value)
        
        return x
    
class Mapper(BaseEstimator,TransformerMixin):
    #
    # ordinal categorical variable mapper:
    # treat column as Ordinal Categorical variable, and assign values accordingly
    #
    def __init__(self,variable:str,mapping:dict):
        
        if not isinstance(variable,str):
            raise ValueError("variable should be a string")
        
        self.variable = variable 
        self.mapping = mapping
    
    def fit (self,X:pd.DataFrame,y:pd.Series = None):
        return self
    
    def transform(self,X:pd.DataFrame)->pd.DataFrame:
        
        x=X.copy()
        x[self.variable] = x[self.variable].map(self.mapping).astype(int)
        
        return x
    
class OutlierHandler(BaseEstimator,TransformerMixin):
    """change the outlier values:
    replace the outlier values with the median of the column
        - to upper-bound , if the value is higher that upper-bound, or 
        - to lower-bound, if the value is lower than lower-bound respectively.
    """
    
    def __init__(self,variable:str):
        if not isinstance(variable,str):
            raise ValuesError("variable name should be a string")
        
        self.variable = variable
    
    def fit(self, X:pd.DataFrame, y:pd.Series = None):
        x=X.copy()
        q1 = x.describe()[self.variable].loc['25%']
        q3 = x.describe()[self.variable].loc['75%']
        
        iqr = q3-q1
        self.lower_bound = q1 - 1.5*iqr
        self.upper_bound = q3 + 1.5*iqr
        
        return self
    
    def transform(self, X:pd.DataFrame)->pd.DataFrame:
        x=X.copy()
        
        for i in x.index:
            if x.loc[i, self.variable]> self.upper_bound:
                x.loc[i,self.variable]=self.upper_bound
            if x.loc[i,self.variable]<self.lower_bound:
                x.loc[i,self.variable]=self.lower_bound
                
        return X

class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, variable:str):
        if not isinstance(variable,str):
            raise ValueError("variable should be a string")
        self.variable = variable
        self.encoder = OneHotEncoder(drop='first',sparse=False)
        
    def  fit(self,X:pd.DataFrame, y: pd.Series =None):
        x = X.copy()
        self.encoder.fit(x[[self.variable]])
        self.encoded_features_name = self.encoder.get_feature_names_out([self.variable])
        return self
    
    def transform(self, X:pd.DataFrame)->pd.DataFrame:
        x=X.copy()
        
        encoded_weekdays = self.encoder.transform(x[[self.variable]])
        
        x[self.encoded_features_name] = encoded_weekdays
        
        x.drop(self.variable, axis = 1, inplace = True)
        
        return x