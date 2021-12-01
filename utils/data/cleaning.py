# libs
import os
import pandas as pd
import numpy as np
from dotenv import find_dotenv, load_dotenv

def check_dtypes(df: pd.DataFrame) -> dict:
    """Checks the column types

    Args:
        df (pd.DataFrame): A pandas dataframe to be analysed

    Returns:
        column_types (dict): Dictionary in which keys are column types
                             and values are lists containing the name of columns
    """
    column_types = {}
    series_types = df.dtypes
    for col, type_ in series_types.iteritems():
        try:
            column_types[str(type_)].append(col)
        except:
            column_types[str(type_)] = [col]

    return column_types

def cast_columns(df:pd.DataFrame, casting:dict) -> pd.DataFrame:
    """Cast columns of a data frame to specific format

    Args:
        df (pd.DataFrame): Original data frame
        casting (dict): Dictionary in which keys are columns to be casted and values are the new types

    Returns:
        df (pd.DataFrame): Transformed dataframe
    """
    for col, type in casting.items():
        df[col] = df[col].astype(type)
    return df

def check_missing(df:pd.DataFrame, columns:list=None) -> pd.Series:
    """Check missing values rate in specified columns of a dataframe

    Args:
        df (pd.DataFrame): A pandas dataframe to be analysed
        columns (list, optional): List of columns to be analysed

    Returns:
        series_missing (pd.Series): An ordered pandas series containing the missing values rate per column
    """
    if columns is None:
        series_missing = (df.isna().sum()/len(df))
        series_missing = series_missing.sort_values(ascending=False)
    else:
        series_missing = (df[columns].isna().sum()/len(df))
        series_missing = series_missing.sort_values(ascending=False)
    return series_missing

def check_constant_columns(df:pd.DataFrame, threshold:float=None):
    """[summary]

    Args:
        df (pd.DataFrame): 
        threshold (float, optional): A threshold number to compare the rate of most frequent element in a column. 
                                     Columns which most frequent element is greater than this threshold will be considered as quasi-constant. Defaults to None.

    Returns:
        [type]: [description]
    """
    constant_columns = []
    quasi_constant_columns = []
    
    for i in df.columns:
        if len(df[i].unique()) == 1:
            df.drop(i, axis = 1, inplace=True)
            constant_columns.append(i)
        else:
            continue 
    if len(constant_columns) > 0:
        print(f'{len(constant_columns)} Constant Columns Found')
        print(constant_columns)
    else:
        print('No Constant Column Found')
        
    if threshold is not None:
        for i in df.columns:
            series_value_counts = df[i].value_counts(normalize=True)
            if series_value_counts.iloc[0] > threshold:
                df.drop(i, axis=1, inplace=True)
                quasi_constant_columns.append(i)
            else:
                continue
        if len(quasi_constant_columns)> 0:
            print(f'{len(quasi_constant_columns)} Quasi-Constant Columns Found')
            print(quasi_constant_columns)
        else:
            print('No Quasi-Constant Column Found')
        return constant_columns, quasi_constant_columns

    else:
        return constant_columns
def check_rare_levels(df:pd.DataFrame, columns:list=None) -> dict:
    """Check the rarer level within each categorical column of a dataframe

    Args:
        df (pd.DataFrame): A pandas dataframe
        columns (list, optional): List of categorical columns to be analysed. 
                                  If List is None, algorithm will try to figure out which columns are categorical. Defaults to None. 

    Returns:
        columns_levels_frequency (dict): Dictionary which keys are column names and values are their most infrequent levels
    """
    columns_levels_frequency = {}
    
    if columns is None:
        columns = df.select_dtypes(include=['O']).columns.tolist()
    for i in columns:
        frequency = df[i].value_counts(normalize=True).sort_values().iloc[0]
        columns_levels_frequency[i] = frequency

    return columns_levels_frequency
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
