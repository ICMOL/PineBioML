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

    def __init__(self, center=True, scale=True, log_domain=False):
        """

        Args:
            center (bool, optional): Pass to Normalizer. Defaults to True.
            scale (bool, optional): Pass to Normalizer. Defaults to True.
            log_domain (bool, optional): Whether input data is in log_domain. Defaults to False.
        """
        self.kernels = {
            #"id3": DT_selection(strategy = "id3", center = center, scale = scale),
            "c45":
            DT_selection(strategy="c45", center=center, scale=scale),
            "RF_gini":
            RF_selection(strategy="gini", center=center, scale=scale),
            #"RF_entropy": RF_selection(strategy = "entropy", center = center, scale = scale),
            #"pcRF_permutation": pcRF_selection(center = center, scale = scale),
            #"pcRF_entropy": pcRF_selection(strategy = "entropy", center = center, scale = scale),
            #"pcRF_permute": pcRF_selection(strategy = "permutation", center = center, scale = scale),
            "AdaBoost":
            AdaBoost_selection(center=center, scale=scale),
            #"Lasso": Lasso_selection(center=center, scale=scale),
            "Lasso_Bisection":
            Lasso_bisection_selection(center=center, scale=scale),
            "multi_Lasso":
            multi_Lasso_selection(center=center, scale=scale),
            "SVM":
            SVM_selection(center=center, scale=scale),
            #"multi_SVM": multi_SVM_selection(center = center, scale = scale),
            #"Volcano_p":
            #Volcano_selection(strategy="p",
            #                 center=center,
            #                scale=scale,
            #               log_domain=log_domain),
            #"Volcano_fold":
            #Volcano_selection(strategy="fold",
            #                  center=center,
            #                  scale=scale,
            #                  log_domain=log_domain),
            "XGboost":
            XGboost_selection(center=center, scale=scale),
            #"Lightgbm":
            #Lightgbm_selection(center=center, scale=scale)
        }

    def Select(self, x, y, k):
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
            results.append(self.kernels[method].Select(x.copy(), y, k))
            print(method, " is done.\n")

        name = pd.concat([pd.Series(i.index, name=i.name) for i in results],
                         axis=1)
        importance = pd.concat(results, axis=1)
        return name, importance

    def plotting(self):
        for method in self.kernels:
            self.kernels[method].plotting()
