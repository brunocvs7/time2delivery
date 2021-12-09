import pandas as pd
import numpy as np
import scipy.stats as st 
from sklearn.pipeline import Pipeline

def get_intervals(model:Pipeline, X:pd.DataFrame, confidence:float=0.95):
    """Gets Prediction Intervals 

    Args:
        model (Pipeline): Pipeline or Model (instance of NGBoost)
        X (pd.DataFrame): Features model will use.
        confidence (float, optional): Confidence Level of interval. Defaults to 0.95.
    """
    X_processed = model['preprocessor'].transform(X)
    y_dist = model['model'].pred_dist(X_processed).params
    predictions = pd.DataFrame(y_dist)
    predictions['interval'] = predictions.apply(lambda x: st.lognorm.interval(alpha=confidence, s=x['s'], scale=x['scale']), axis=1)
    return predictions