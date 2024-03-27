import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from .base import selection as base_selection

class SVM_selection(base_selection):
    def __init__(self, center = True, scale = False, log_transform = True):
        super().__init__(center = center, scale = scale, log_transform = log_transform)
        self.kernel = LinearSVC(dual="auto", class_weight="balanced")

    def scoring(self, x, y = None):
        self.kernel.fit(x, y)
        svm_weights = np.abs(self.kernel.coef_).sum(axis=0)
        svm_weights /= svm_weights.sum()
        
        return pd.Series(svm_weights, index = x.columns, name = "SVM")

    def choose(self, score, k):
        return score.sort_values().tail(k)

class multi_SVM_selection(base_selection):
    def __init__(self):
        pass

    def select(self, x, y, k, batch_k = 5):
        result = []
        for i in range(k//batch_k):
            result.append(SVM_selection().select(x, y, batch_k))
            x = x.drop(result[-1].index, axis = 1)
        result = pd.concat(result)
        result.name = "multi_svm"
        return result-result.min()    
            
