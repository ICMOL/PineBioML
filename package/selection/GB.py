from .base import selection as base_selection
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgbm


class XGboost_selection(base_selection):
    def __init__(self, unbalanced= True, center = True, scale = True):
        super().__init__(center = center, scale = scale)
        self.unbalanced = unbalanced
            
        self.kernel = xgb.XGBClassifier()
        self.name = "XGboost"

    def scoring(self, x, y = None):
        if self.unbalanced:
            sample_weight = (y/y.mean() + (1-y)/((1-y).mean()))/2
        else:
            sample_weight = np.ones(len(y))
            
        self.kernel.fit(x, y, sample_weight = sample_weight)
        score = self.kernel.feature_importances_
        self.scores = pd.Series(score, index=x.columns, name = self.name).sort_values(ascending = False)
        return self.scores.copy()

class Lightgbm_selection(base_selection):
    def __init__(self, unbalanced= True, center = True, scale = False):
        super().__init__(center = center, scale = scale)
        self.unbalanced = unbalanced
            
        self.kernel = lgbm.LGBMClassifier()
        self.name = "Lightgbm"

    def scoring(self, x, y = None):
        if self.unbalanced:
            sample_weight = (y/y.mean() + (1-y)/((1-y).mean()))/2
        else:
            sample_weight = np.ones(len(y))
            
        self.kernel.fit(x, y, sample_weight = sample_weight)
        score = self.kernel.feature_importances_
        self.scores = pd.Series(score, index=x.columns, name = self.name).sort_values(ascending = False)
        return self.scores.copy()


if __name__ == "__main__":
    XGboost_selection()
