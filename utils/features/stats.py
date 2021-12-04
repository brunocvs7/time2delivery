from numpy.lib.arraysetops import unique
import pandas as pd
from scipy.stats import mstats

def check_kruskal_wallis(df:pd.DataFrame, x:str, y:str, alpha:float=0.05):
    """Applies Kruskal-Wallis test on a list of arrays

    Args:
        df (pd.DataFrame): A pandas dataframe with the original data
        x (str): Name of the column contaning the groups to be tested
        y (str): Name of the column with the characteristic to be tested
        alpha (float, optional): Significante level to reject Null hypothesis. Defaults to 0.05.
    """
    unique_groups = df[x].unique()
    list_arrays = []

    for i in unique_groups:
        list_arrays.append(df[df[x]==i][y].values)
    H, pvalue = mstats.kruskalwallis(*list_arrays)
    print("H-statistic:", H)
    print("P-Value:", pvalue)

    if pvalue < alpha:
        print("Reject NULL hypothesis - Significant differences exist between groups.")
    if pvalue > alpha:
        print("Accept NULL hypothesis - No significant difference between groups.")
        

