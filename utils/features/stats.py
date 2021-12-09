from numpy.lib.arraysetops import unique
import pandas as pd
from scipy.stats import mstats

def check_kruskal_wallis(df:pd.DataFrame, cat_columns:list, target:str, alpha:float=0.05):
    """Applies Kruskal-Wallis test on a list of arrays

    Args:
        df (pd.DataFrame): A pandas dataframe with the original data
        x (str): Name of the column contaning the groups to be tested
        y (str): Name of the column with the characteristic to be tested
        alpha (float, optional): Significance level to reject Null hypothesis. Defaults to 0.05.
    Returns:
        kw_result: A dictionary, in which keys are name of columns and values are flags indicating if the feature passed the test or not.
    """
    kw_result = dict()
    for col in cat_columns:
        unique_groups = df[col].unique()
        list_arrays = []

        for i in unique_groups:
            list_arrays.append(df[df[col]==i][target].values)
        H, pvalue = mstats.kruskalwallis(*list_arrays)
        if pvalue < alpha:
            kw_result[col] = 'reject'
        if pvalue > alpha:
            kw_result[col] = 'not reject'
    return kw_result

def check_spearman_correlation(df:pd.DataFrame, num_columns:list, target, alpha:float=0.05):
    """Performs spearman correlation test

    Args:
        df (pd.DataFrame): A dataframe containing features and target
        num_columns (list): List of numerical columns to be tested
        target ([type]): Name of the target column
        alpha (float, optional): Significance level for the test. Defaults to 0.05.

    Returns:
        sp_result: A dictionary, in which keys are name of columns and values are flags indicating if the feature passed the test or not.
    """
    sp_result = dict()
    for col in num_columns:
        correlation, pvalue = mstats.spearmanr(df[col], df[target])
        if pvalue < alpha:
            sp_result[col] = 'reject'
        if pvalue > alpha:
           sp_result[col] = 'not reject'
    return sp_result

