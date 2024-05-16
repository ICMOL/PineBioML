from .base import selection as base_selection
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

class RF_selection(base_selection):
    def __init__(self, trees = 1024*16, unbalanced = True, strategy = "gini", center = True, scale = False):
        super().__init__(center = center, scale = scale)
        self.strategy = strategy
        if unbalanced:
            class_weight = "balanced"
        else:
            class_weight = None
            
        self.kernel = RandomForestClassifier(n_estimators = trees, n_jobs=-1, max_samples = 0.75, class_weight = class_weight, criterion = strategy, verbose = 1)
        self.name = "RandomForest_"+self.strategy

    def scoring(self, x, y = None):
        self.kernel.fit(x, y)
        score = self.kernel.feature_importances_
        self.scores = pd.Series(score, index=x.columns, name = self.name).sort_values()
        return self.scores.copy()
    
class AdaBoost_selection(base_selection):
    def __init__(self, unbalanced = True, n_iter = 64, learning_rate = 0.1, center = True, scale = False):
        super().__init__(center = center, scale = scale)
        self.unbalanced = unbalanced
        self.kernel = AdaBoostClassifier(n_estimators = n_iter, learning_rate= learning_rate)
        self.name = "AdaBoost"+str(n_iter)

    def scoring(self, x, y = None):
        print("I don't have a progress bar but I am running")
        if self.unbalanced:
            sample_weight = (y/y.mean() + (1-y)/((1-y).mean()))/2
        else:
            sample_weight = np.ones(len(y))
        self.kernel.fit(x, y, sample_weight)
        score = self.kernel.feature_importances_
        self.scores = pd.Series(score, index=x.columns, name = self.name).sort_values()
        return self.scores.copy()

if __name__ == "__main__":
    RF_selection()
