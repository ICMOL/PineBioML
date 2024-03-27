from .base import selection as selection_base
from .DT import DT_selection
from .RF import RF_selection
from .Volcano import Volcano_selection
from .Lasso import Lasso_selection, Lasso_bisection_selection, multi_Lasso_selection
from .SVM import SVM_selection, multi_SVM_selection
import pandas as pd

class selector(selection_base):
    def __init__(self):
        self.kernels = {
            "id3": DT_selection(strategy = "id3"),
            "c45": DT_selection(strategy = "c45"),
            "RF_gini": RF_selection(strategy = "gini"),
            "RF_etropy": RF_selection(strategy = "entropy"),
            "Lasso": Lasso_selection(),
            #"Lasso": Lasso_bisection_selection(), 
            "multi_Lasso": multi_Lasso_selection(),
            "SVM": SVM_selection(),
            "multi_SVM": multi_SVM_selection(),
            "Volcano_p": Volcano_selection(strategy = "p"), 
            "Volcano_fold":Volcano_selection(strategy = "fold"), 
            }
        
    def select(self, x, y, k):
        results = []
        for method in self.kernels:
            print("Using ", method, " to select.")
            results.append(self.kernels[method].select(x.copy(), y, k))
            print(method, " is done.")

        name = pd.concat([pd.Series(i.index, name = i.name) for i in results], axis = 1)
        importance = pd.concat(results, axis = 1)
        return name, importance

if __name__ == "__main__":
    selector()
