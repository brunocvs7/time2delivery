import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer 
from sklearn.model_selection import KFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

class Boruta:
    """
    A class to perform feature selection, based on BorutaPy Class of boruta package
    This version is based only on feature importance of a random forest model and returns results more pretifully
    See https://github.com/scikit-learn-contrib/boruta_py for more details (original implementation)

    ...

    Attributes
    ----------
    n_iter : int
        number of iterations the algorithm will perform
    columns_removed : list 
        list of columns to be removed (Obtained after fit method runs)
 

    Methods
    -------
    fit(X, y):
        Runs Boruta Algorithm. It brings a list of columns We should remove and a boolean vetor.
    """

    def __init__(self, n_iter=100):
        """
        Constructs all the necessary attributes for the boruta object.

        Parameters
        ----------
        n_iter : int
            number of iterations the algorithm will perform
        """
        self.n_iter = n_iter
        self._columns_remove_boruta = None
        self._bool_decision = None
        self._best_features = None

    def fit(self, X, y, cat_columns=True, num_columns=True):
        """
        Runs Boruta Algorithm.

        Parameters
        ----------
        X : pandas.dataframe
            Pandas Data Frame with all features
        y: pandas.dataframe
            Pandas Data Frame with target
    
        Returns
        -------
        None
        """
        X.replace(to_replace=[None], value=np.nan, inplace=True)
        if (num_columns == False) & (cat_columns == True):
            cat_columns = X.select_dtypes(include=['object']).columns.tolist()
            X.loc[:, cat_columns] = X.loc[:, cat_columns].astype('str')
            cat_pipe_preprocessor = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'most_frequent')), ('cat_transformer', OrdinalEncoder())])
            preprocessor = ColumnTransformer(transformers = [('cat_pipe_preprocessor', cat_pipe_preprocessor, cat_columns)])
            X_processed = preprocessor.fit_transform(X)
            rf = RandomForestRegressor(n_jobs=-1, max_depth=5, random_state=123)    
            # Criando o boruta
            selector = BorutaPy(rf, n_estimators='auto',random_state=123, max_iter = self.n_iter) 
            selector.fit(X,y)
        elif (cat_columns==False) &  (num_columns==True):
            num_columns = X.select_dtypes(include=['int','float', 'int64', 'float64', 'int32', 'float32']).columns.tolist() 
            num_pipe_preprocessor = Pipeline(steps= [('imputer',SimpleImputer(strategy = 'median'))])
            preprocessor = ColumnTransformer(transformers = [('num_pipe_preprocessor',num_pipe_preprocessor, num_columns)])
            X_processed = preprocessor.fit_transform(X)
            rf = RandomForestRegressor(n_jobs=-1, max_depth=5, random_state=123)    
            # Criando o boruta
            selector = BorutaPy(rf, n_estimators='auto',random_state=123, max_iter = self.n_iter) 
            selector.fit(X_processed,y)
        else:     
            cat_columns = X.select_dtypes(include=['object']).columns.tolist()
            X.loc[:, cat_columns] = X.loc[:, cat_columns].astype('str')
            num_columns = X.select_dtypes(include=['int','float', 'int64', 'float64', 'int32', 'float32']).columns.tolist() 
            num_pipe_preprocessor = Pipeline(steps= [('imputer',SimpleImputer(strategy = 'median'))])
            cat_pipe_preprocessor = Pipeline(steps = [('imputer', SimpleImputer(strategy = 'most_frequent')), ('cat_transformer', OrdinalEncoder())])
            preprocessor = ColumnTransformer(transformers = [('num_pipe_preprocessor',num_pipe_preprocessor, num_columns), ('cat_pipe_preprocessor', cat_pipe_preprocessor, cat_columns)])
            X_processed = preprocessor.fit_transform(X)
            rf = RandomForestRegressor(n_jobs=-1, max_depth=5, random_state=123)    
            # Criando o boruta
            selector = BorutaPy(rf, n_estimators='auto',random_state=123, max_iter = self.n_iter) 
            selector.fit(X_processed,y)
        features_accept = X.columns[selector.support_].to_list()
        features_irresolution = X.columns[selector.support_weak_].to_list()
        all_columns = set(X.columns)
        keep_columns = set(features_accept+features_irresolution)
        features_to_drop = all_columns.difference(keep_columns)

        self._columns_indecisive = features_irresolution
        self._columns_remove_boruta = features_to_drop
        self._columns_best_features = features_accept
        