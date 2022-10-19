"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.18.3
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from typing import Any, Dict, Tuple
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import mlflow

def getQuartiles(dataFrame, column):
    Q1 = np.percentile(dataFrame[column].dropna(), 25, interpolation = 'midpoint')
    Q3 = np.percentile(dataFrame[column].dropna(), 75, interpolation = 'midpoint')

    IQR = Q3 - Q1

    return {'Q1': Q1,'Q3': Q3, 'IQR': IQR}

def getBounds(dataFrame, column, quartiles):
    Q1 = quartiles['Q1']
    Q3 = quartiles['Q3']
    IQR = quartiles['IQR']

    #print('Q1', Q1)
    #print('Q3', Q3)
    #print('IQR', IQR)

    # Upper bound
    upper = np.where(dataFrame[column] >= (Q3 + 1.5*IQR))

    # Below Lower bound
    lower = np.where(dataFrame[column] <= (Q1 - 1.5*IQR))

    return {'upper': upper, 'lower': lower}

def deleteOutliersDataframe(dataFrame, bounds):
    upper = bounds['upper']
    lower = bounds['lower']

    dataFrame.drop(upper[0], inplace = True)
    dataFrame.drop(lower[0], inplace = True)
    dataFrame.reset_index(inplace = True, drop = True)

    return dataFrame

def deleteOutliersSetTrain(train, column):
    quartiles = getQuartiles(train.copy(), column)
    bounds = getBounds(train.copy(), column, quartiles)
    train = deleteOutliersDataframe(train.copy(), bounds)

    return train

# funtion to remove columns from data
def drop_cols(data: pd.DataFrame, drop_cols: list = None) -> pd.DataFrame:
    """Drop columns from dataframe"""
    return data.drop(drop_cols, axis=1, inplace = True)

# function to replace np.nan values with '?'
def replace_unknown_values(data: pd.DataFrame, replace_values: np.nan) -> pd.DataFrame:
    """Replace unknown values with '?'"""
    return data.replace(replace_values, '?', inplace = True)

def drop_exact_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate rows from data."""
    return data.drop_duplicates(keep=False)

def drop_outliers(data: pd.DataFrame, col: str) -> pd.DataFrame:
    data = deleteOutliersSetTrain(data,col)
    return data

def fill_nan(data: pd.DataFrame, col:str,value: float ) -> pd.DataFrame:
    data[col].replace(np.nan,value,inplace=True)
    return data

def drop_nan(data: pd.DataFrame, col:str) -> pd.DataFrame:
    data.drop(data[data[col].notnull() == False].index, inplace=True)
    data.reset_index(inplace = True, drop = True)
    return data

def drop_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    data.drop_duplicates(inplace=True)
    return data

def reset_index(data: pd.DataFrame) -> pd.DataFrame:
    data.reset_index(inplace = True, drop = True)
    return data


def first_processing(data: pd.DataFrame, parameters: Dict[str, Any]) -> pd.DataFrame:

    pipe_functions = [
        ('buil_features',FunctionTransformer(buil_features))
    ]
    pipe_line_train_data = Pipeline(steps=pipe_functions)
    return data, ('first_processing',pipe_line_train_data)

def data_type_splits(parameters: Dict[str, Any]):
    coll_fill_mode = parameters['coll_fill_mode']
    coll_fill_mean = parameters['coll_fill_mean']
    return coll_fill_mode, coll_fill_mean

def fill_nan_mode(coll_fill_mode):

    pipe_functions = [
        ('imputer_mode',SimpleImputer (strategy='most_frequent',missing_values=np.nan))
    ]
    fill_nan_mode_pipe =  Pipeline(steps=pipe_functions)
    return ('fill_mode',fill_nan_mode_pipe, coll_fill_mode)

def fill_nan_mean(coll_fill_mean):

    pipe_functions = [
        ('imputer_mean',SimpleImputer(strategy='mean',missing_values=np.nan))
    ]
    fill_nan_mean_pipe =  Pipeline(steps=pipe_functions)
    return ('fill_mean',fill_nan_mean_pipe, coll_fill_mean)




def buil_features(data: pd.DataFrame) -> pd.DataFrame:
    
    data = (data
                    .pipe(drop_outliers, col ="crim")    
                    #.pipe(fill_nan, col= "crim",value= 1.27038)
                    #.pipe(fill_nan, col= "zn",value= 0.0) 
                    .pipe(drop_nan, col= "indus") 
                    #.pipe(fill_nan, col= "chas", value= 0.0) 
                    .pipe(drop_outliers, col ="nox") 
                    .pipe(drop_nan, col= "nox") 
                    .pipe(drop_outliers, col ="rm") 
                    .pipe(drop_nan, col= "rm") 
                    .pipe(drop_nan, col= "age") 
                    #.pipe(fill_nan, col= "dis", value= 4.2820) 
                    .pipe(drop_outliers, col ="dis") 
                    #.pipe(fill_nan, col= "rad", value= 4.0)
                    #.pipe(fill_nan, col= "tax", value= 666.0)
                    #.pipe(fill_nan, col= "ptratio", value= 20.2)
                    #.pipe(fill_nan, col= "black", value= 396.9)
                    .pipe(drop_outliers, col ="black")
                    #.pipe(fill_nan, col= "lstat", value= 1.73)
                    .pipe(drop_outliers, col ="lstat")
                    .pipe(drop_nan, col= "lstat") 
                    #.pipe(drop_nan, col= "medv") 
                    .pipe(reset_index)
                    )
    return data

def last_processing(x: pd.DataFrame,
                   y:pd.Series,
                    first: tuple,
                    coll_fill_mode: Tuple,
                    coll_fill_mean: Tuple ) -> Pipeline:

    data = pd.concat([x,y],axis=1)

    pipe_transforms = Pipeline(steps= [
        first,
        ('columns', ColumnTransformer(
                        transformers=[
                            coll_fill_mode,
                            coll_fill_mean,
                        ],
                        remainder='passthrough')
         ) 
        ])

    data_transformed = pipe_transforms.fit_transform(data)

    mlflow.set_experiment('house-pricing')
    mlflow.log_param(f"shape train_transformed", data_transformed.shape)

    return pipe_transforms, data_transformed

def post_processing(x_in: np.ndarray) -> np.ndarray:
    """
    General processing to transformed data, like remove duplicates
    important after transformation the data types are numpy ndarray
    Args:
        x_in: x data after transformations
        y_train: y_train
    Returns:
    """
    methods = ["remove duplicates"]
    mlflow.set_experiment('house-pricing')
    mlflow.log_param('post-processing', methods)

    df = pd.DataFrame(x_in)
    df.drop_duplicates()

    x_out = df.iloc[:, :-1]
    y_out = df.iloc[:, -1]

    #print(df.info())
    #print("medv")
    #print(df.iloc[:, -1])
    #print(x_out)

    mlflow.log_param('shape post-processing', x_out.shape)
    return x_out, y_out


