import pandas as pd

class selection:
    def __init__(self):
        pass

    def select(self, x, y, k):
        # x should be a pd dataframe or a numpy array without missing value
        
        score = self.scoring(x, y)
        return self.choose(score, k)

    def scoring(self, x, y = None):
        return x.max()

    def choose(self, score, k):
        return score.sort_values().tail(k)