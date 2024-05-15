import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from .base import selection as base_selection

class RF_selection(base_selection):
    def __init__(self, plotting = False, trees = 1024*32, unbalanced = True, strategy = "gini", center = True, scale = False):
        super().__init__(center = center, scale = scale)
        self.strategy = strategy
        if unbalanced:
            class_weight = "balanced"
        else:
            class_weight = None
            
        self.kernel = RandomForestClassifier(n_estimators = trees, n_jobs=-1, max_samples = 0.75, class_weight = class_weight, criterion = strategy)

    def scoring(self, x, y = None):
        print("I don't have a progress bar but I am running")
        self.kernel.fit(x, y)
        score = self.kernel.feature_importances_
        score = pd.Series(score[score>0], index=x.columns[score>0], name = "RandomForest_"+self.strategy)
        return score

    def choose(self, score, k):
        return score.sort_values().tail(k)

if __name__ == "__main__":
    RF_selection()
