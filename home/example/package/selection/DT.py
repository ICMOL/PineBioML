import pandas as pd
import numpy as np
from tqdm import tqdm
from .base import selection as base_selection


class DT_selection(base_selection):
    def __init__(self, bins = 10, q = 0.05, strategy = "c45", center = True, scale = False, log_transform = True):
        super().__init__(center = False, scale = scale, log_transform = log_transform)
        self.bins = bins -1
        self.q = q
        self.strategy = strategy

    def scoring(self, x, y = None):
        upper = x.quantile(1-self.q)
        lower = x.quantile(self.q)
        #print(upper, lower)
        normed = (x-lower)/(upper-lower)
        normed = normed.clip(0, 1)
        bin_idx = (normed*self.bins - 0.5).round().astype(np.int32)
        columns = bin_idx.columns
        
        bin_idx["label"] = y
        
        scores = []
        for i in tqdm(columns):
            feature_hists = bin_idx[[i, "label"]].groupby(i)
            feature_entropy = feature_hists.apply(self.entropy)
            feature_size = feature_hists.apply(len)+1e-3
            
            info = (feature_entropy/feature_size).sum()
            gain = 0 - info
            if self.strategy == "c45":
                freq = bin_idx[i].value_counts()
                p = freq/freq.sum()
                split_info = -p*np.log(p)
                gain /= split_info.sum()
            scores.append(gain)
        scores = pd.Series(scores, index = columns, name = "DT_score_"+ self.strategy)
        return scores

    def choose(self, score, k):
        score = score+ score.min()
        return score.sort_values().tail(k)

    def entropy(self, x):
        x = x["label"]
        p = x.mean()
        q = 1-p
        if p and q: # not all zero
            return -(p*np.log(p+ 1e-6)+q*np.log(q+ 1e-6))
        else:
            return 0
