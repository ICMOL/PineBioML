import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from .base import selection as base_selection

class SVM_selection(base_selection):
    def __init__(self, center = True, scale = False):
        super().__init__(center = center, scale = scale, global_scale = True)
        self.kernel = LinearSVC(dual="auto", class_weight="balanced")
        self.name = "SVM"

    def scoring(self, x, y = None):
        self.kernel.fit(x, y)
        svm_weights = np.abs(self.kernel.coef_).sum(axis=0)
        svm_weights /= svm_weights.sum()
        
        self.scores = pd.Series(svm_weights, index = x.columns, name = self.name).sort_values()
        return self.scores.copy()
    

class multi_SVM_selection(base_selection):
    def __init__(self, center = True, scale = True):
        self.center = center
        self.scale = scale
        self.name = "multi_svm"

    def select(self, x, y, k, batch_k = 5):
        result = []
        if k == -1:
            k = x.shape[0]
        for i in range(k//batch_k):
            result.append(SVM_selection(center = self.center, scale = self.scale).select(x, y, batch_k))
            x = x.drop(result[-1].index, axis = 1)
        
        result = pd.concat(result)
        result = result- result.min()
        result.name = self.name

        self.selected_score = result.sort_values()
        return self.selected_score.copy()
            
