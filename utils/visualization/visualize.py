import scipy
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch

def cluster_correlation(corr_array:pd.DataFrame, inplace:bool=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to eachother 
    
    Args:
        corr_array (pd.DataFrame): Pandas dataframe or numpy.ndarray
                     a NxN correlation matrix 
        inplace (bool): Boolean flag. If True it modifies original dataframe passed.
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    
    if not inplace:
        corr_array = corr_array.copy()
    
    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx]