import pandas as pd
import numpy as np

class selection:
    def __init__(self, center = True, scale = True, log_transform = False):
        self.normalizer = normalizer(center = center, scale = scale, log_transform = log_transform)
        

    def select(self, x, y, k):
        # x should be a pd dataframe or a numpy array without missing value
        x, y = self.normalizer.fit_transform(x, y)
        score = self.scoring(x, y)
        return self.choose(score, k)

    def scoring(self, x, y = None):
        return x.max()

    def choose(self, score, k):
        return score.sort_values().tail(k)
    

class normalizer:
    def __init__(self, center = True, scale = True, log_transform = False):
        self.center = center
        self.mean = 0
        self.scale = scale
        self.norm = 1
        self.log_transform = log_transform
        self.fitted = False

    def fit(self, x, y = None):
        if self.log_transform:
            x = np.log(x)
        if self.center:
            self.mean = x.mean()
        if self.scale:
            self.norm = x.std()
        self.fitted = True
        return self
    
    def transform(self, x, y = None):
        if not self.fitted:
            print("WARNING: please call fit before calling transform")
        if self.log_transform:
            x = np.log(x)
        if self.center:
            x = x - self.mean
        if self.scale:
            x = x / self.norm
        return x, y

    def fit_transform(self, x, y = None):
        self.fit(x, y)
        return self.transform(x, y)

    def inverse_transform(self, x, y = None):
        if self.scale:
            x = x * self.norm
        if self.center:
            x = x + self.mean
        if self.log_transform:
            x = np.exp(x)

        return x, y