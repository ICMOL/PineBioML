import pandas as pd
import numpy as np
from tqdm import tqdm
from . import SelectionPipeline


class DT_selection(SelectionPipeline):
    """
    A child class of SelectionPipeline .

    Using Decision stump (a single Decision tree) to scoring features.
    """

    def __init__(self, k, bins=10, q=0.05, strategy="c45"):
        """
        Args:
            bins (int, optional): Bins to esimate data distribution entropy. Defaults to 10.
            q (float, optional): Clip data values out of [q, 1-q] percentile to reduce the affect of outliers while estimate entropy. Defaults to 0.05.
            strategy (str, optional): One of {"id3", "c45"}. The strategy to build decision tree. Defaults to "c45".
        """
        super().__init__(k=k)
        self.bins = bins - 1
        self.q = q
        self.strategy = strategy
        self.name = "DT_score_" + self.strategy

    def Scoring(self, x, y=None):
        """
        Using Decision stump (a single Decision tree) to scoring features. Though, single layer stump is equivalent to compare the id3/c4.5 score directly.

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.

        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.
        """
        upper = x.quantile(1 - self.q)
        lower = x.quantile(self.q)
        #print(upper, lower)
        normed = (x - lower) / (upper - lower)
        normed = normed.clip(0, 1)
        bin_idx = (normed * self.bins - 0.5).round().astype(np.int32)
        columns = bin_idx.columns

        bin_idx["label"] = y

        scores = []
        for i in tqdm(columns):
            feature_hists = bin_idx[[i, "label"]].groupby(i)
            feature_entropy = feature_hists.apply(self.entropy)
            feature_size = feature_hists.apply(len) + 1e-3

            info = (feature_entropy / feature_size).sum()
            gain = 0 - info
            if self.strategy == "c45":
                freq = bin_idx[i].value_counts()
                p = freq / freq.sum()
                split_info = -p * np.log(p)
                gain /= split_info.sum()
            scores.append(gain)
        scores = pd.Series(scores, index=columns,
                           name=self.name).sort_values(ascending=False)
        scores = scores - scores.min()
        return scores

    def entropy(self, x):
        """
        Estimate entropy

        Args:
            x (pandas.DataFrame): data with bined label

        Returns:
            float: entropy
        """
        label_nums = x["label"].value_counts()
        label_prop = label_nums / label_nums.sum()

        entropy = -(label_prop * np.log(label_prop + 1e-6)).sum()

        return entropy
