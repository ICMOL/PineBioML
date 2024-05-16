from .base import selection as selection_base
from .DT import DT_selection
from .RF import RF_selection, AdaBoost_selection
from .Volcano import Volcano_selection
from .Lasso import Lasso_selection, Lasso_bisection_selection, multi_Lasso_selection
from .SVM import SVM_selection, multi_SVM_selection
from .GB import XGboost_selection, Lightgbm_selection
import pandas as pd

class selector(selection_base):
    def __init__(self, center = True, scale = True, log_domain = False):
        self.kernels = {
            "id3": DT_selection(strategy = "id3", center = center, scale = scale),
            "c45": DT_selection(strategy = "c45", center = center, scale = scale),
            "RF_gini": RF_selection(strategy = "gini", center = center, scale = scale),
            "RF_etropy": RF_selection(strategy = "entropy", center = center, scale = scale),
            "AdaBoost": AdaBoost_selection(center = center, scale = scale),
            "Lasso": Lasso_selection(center = center, scale = scale),
            #"Lasso_Bisection": Lasso_bisection_selection(center = center, scale = scale), 
            "multi_Lasso": multi_Lasso_selection(center = center, scale = scale),
            "SVM": SVM_selection(center = center, scale = scale),
            "multi_SVM": multi_SVM_selection(center = center, scale = scale),
            "Volcano_p": Volcano_selection(strategy = "p", center = center, scale = scale, log_domain=log_domain), 
            "Volcano_fold":Volcano_selection(strategy = "fold", center = center, scale = scale, log_domain=log_domain), 
            "XGboost": XGboost_selection(center = center, scale = scale),
            "Lightgbm": Lightgbm_selection(center = center, scale = scale)
            }
        
    def select(self, x, y, k):
        results = []
        for method in self.kernels:
            print("Using ", method, " to select.")
            results.append(self.kernels[method].select(x.copy(), y, k))
            print(method, " is done.\n")

        name = pd.concat([pd.Series(i.index, name = i.name) for i in results], axis = 1)
        importance = pd.concat(results, axis = 1)
        return name, importance
    
    def plotting(self):
        for method in self.kernels:
            self.kernels[method].plotting()


if __name__ == "__main__":
    selector()
