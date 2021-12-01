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
        columns (list): List of columns to be analysed (optional)

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

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
