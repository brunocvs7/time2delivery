import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import learning_curve
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


def plot():
    pass

def plot_permutation_importance(model:Pipeline, X:pd.DataFrame, y:pd.DataFrame, columns, n_repeats:int=5, scoring:str='average_precision', n_best:int=10):
    """Plots permutation importances

    Args:
        model (Pipeline): A sklearn model or pipeline.
        X (pd.DataFrame): Features.
        y (pd.DataFrame): Target.
        n_repeats (int, optional): Number of repetitions for the algorithm to run. Defaults to 5.
        scoring (str, optional): Metric to be analysed. Defaults to 'average_precision'.
        n_best (int, optional): Number of best features to be ploted. Defaults to 10.
    """
    columns = columns 
    results_pimp = permutation_importance(model, X, y, n_repeats=n_repeats,
                                random_state=42, scoring=scoring, n_jobs=-1)
    sorted_idx = results_pimp.importances_mean.argsort()  
    fig, ax = plt.subplots(figsize=(10,10))
    labels = [columns[i] for i in sorted_idx][-n_best:]
    ax.boxplot(results_pimp.importances[sorted_idx][-n_best:].T,
            vert=False, labels=labels)
    ax.set_title(f"Permutation Importances - {n_best} best features")
    fig.tight_layout()
    plt.show()


def plot_learning_curve(model:Pipeline, X:pd.DataFrame, y:pd.DataFrame, cv:int, scoring:str='neg_mean_squared_error'):
    """Plots Learning Curve

    Args:
        model (Pipeline): A sklearn Pipeline or Model
        X (pd.DataFrame): Features
        y (pd.DataFrame or pd.Series): Target
        cv (int): Cross validation folds
    """

    train_sizes, train_scores, validation_scores = learning_curve(model, 
                                                                  X, 
                                                                  y,
                                                                  cv=cv, 
                                                                  scoring=scoring)
    train_scores_mean=-train_scores.mean(axis = 1)
    validation_scores_mean=-validation_scores.mean(axis = 1)

    plt.plot(train_sizes, train_scores_mean, label = 'Training error')
    plt.plot(train_sizes, validation_scores_mean, label = 'Validation error')

    plt.ylabel('MSE', fontsize = 14)
    plt.xlabel('Training set size', fontsize = 14)
    title = 'Learning curves for a ' + str(model).split('(')[0] + ' model'
    plt.title(title, fontsize = 18, y = 1.03)
    plt.legend()
    #plt.ylim(0,40)
    plt.show()