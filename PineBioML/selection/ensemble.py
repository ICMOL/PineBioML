from . import SelectionPipeline
from .DT import DT_selection
from .RF import RF_selection, pcRF_selection
from .Volcano import Volcano_selection
from .Lasso import Lasso_selection, multi_Lasso_selection, Lasso_bisection_selection
from .SVM import SVM_selection  #, multi_SVM_selection
from .Boosting import XGboost_selection, Lightgbm_selection, AdaBoost_selection
import pandas as pd


class selector(SelectionPipeline):
    """
    A functional stack of diffirent methods.
    
    """

    def __init__(self, k=-1, RF_trees=1024, z_importance_threshold=0):
        """

        Args:

        """
        self.z_importance_threshold = z_importance_threshold

        self.kernels = {
            "c45": DT_selection(k=k, strategy="c45"),
            "RF_gini": RF_selection(k=k, strategy="gini", trees=RF_trees),
            #"RF_entropy": RF_selection(strategy = "entropy"),
            #"pcRF_permutation": pcRF_selection(),
            #"pcRF_entropy": pcRF_selection(strategy = "entropy"),
            #"pcRF_permute": pcRF_selection(strategy = "permutation"),
            "AdaBoost": AdaBoost_selection(k=k),
            #"Lasso": Lasso_selection(center=center),
            "Lasso_Bisection": Lasso_bisection_selection(k=k),
            "multi_Lasso": multi_Lasso_selection(k=k),
            "SVM": SVM_selection(k=k),
            "XGboost": XGboost_selection(k=k),
            "Lightgbm": Lightgbm_selection(k=k)
        }

    def Select(self, x, y):
        """
        Calling all the methods in kernel sequancially.

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.
            k (int): Number of feature to select. The result may less than k

        Returns:
            pandas.Series: The concatenated results. Top k (may less than k) important feature from diffient methods.
        """
        results = []
        for method in self.kernels:
            print("Using ", method, " to select.")
            results.append(self.kernels[method].Select(x.copy(), y))
            print(method, " is done.\n")

        name = pd.concat([pd.Series(i.index, name=i.name) for i in results],
                         axis=1)
        importance = pd.concat(results, axis=1)
        self.selected_score = importance
        return name, importance

    def fit(self, x, y):
        """
        sklearn api

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods.
        """
        name, importance = self.Select(x, y)

        return self

    def transform(self, x):
        z_scores = (self.selected_score - self.selected_score.mean()) / (
            self.selected_score.std() + 1e-4)
        z_scores = z_scores.mean(axis=1)
        z_scores = z_scores[z_scores > self.z_importance_threshold]

        return x[z_scores.index]

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)

    def plotting(self):
        for method in self.kernels:
            self.kernels[method].plotting()
