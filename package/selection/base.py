import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class selection:
    def __init__(self, center = True, scale = True, global_scale = False):
        self.normalizer = normalizer(center = center, scale = scale, global_scale = global_scale)
        self.scores = None
        self.name = "base"

    def select(self, x, y, k):
        # x should be a pd dataframe or a numpy array without missing value
        x, y = self.normalizer.fit_transform(x, y)
        scores = self.scoring(x, y)
        selected_score = self.choose(scores, k)
        return selected_score

    def scoring(self, x, y = None):
        self.scores = x.max().sort_values(ascending = False) # top is better
        return self.scores.copy()

    def choose(self, scores, k):
        self.selected_score = scores.head(k)
        self.selected_score = self.selected_score[self.selected_score != 0]
        
        return self.selected_score.copy()
    
    def plotting(self):
        fig, ax = plt.subplots(1, 1)
        ax.bar(self.selected_score.index, self.selected_score)
        for label in ax.get_xticklabels(which='major'):
            label.set(rotation=45, horizontalalignment='right')
        ax.set_title(self.name+" score")

        #upper = self.selected_score.max()
        #lower = self.selected_score.min()
        #scale = upper - lower
        #ax.set_ylim((lower - 0.05*scale, upper + 0.05*scale))

        plt.show()

    def diagnosing(self):
        pass

    def report(self):
        pass
        
    

class normalizer:
    def __init__(self, center = True, scale = True, global_scale = False):
        self.center = center
        self.mean = 0
        self.scale = scale
        self.norm = 1
        #self.log_transform = log_transform
        self.global_scale = global_scale
        self.global_norm = 1
        self.fitted = False

    def fit(self, x, y = None):
        #if self.log_transform:
         #   x = np.log(x)
        if self.center:
            self.mean = x.mean()
            #print("mean: ", self.mean)
            x = x - self.mean
        if self.scale:
            self.norm = x.std()
            #print("std: ", self.norm)
            x = x / self.norm
        if self.global_scale:
            self.global_norm = x.values.std()
            #print("global std: ", self.global_norm)
            x = x / self.global_norm
                
        self.fitted = True
        return self
    
    def transform(self, x, y = None):
        if not self.fitted:
            print("WARNING: please call fit before calling transform")
        #if self.log_transform:
         #   x = np.log(x)
        if self.center:
            x = x - self.mean
        if self.scale:
            x = x / self.norm
        if self.global_scale:
            x = x / self.global_norm
        return x, y

    def fit_transform(self, x, y = None):
        self.fit(x, y)
        return self.transform(x, y)

    def inverse_transform(self, x, y = None):
        if self.global_scale:
            x = x * self.global_norm
        if self.scale:
            x = x * self.norm
        if self.center:
            x = x + self.mean
        #if self.log_transform:
         #   x = np.exp(x)
        return x, y