import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_sample_weight


def sample_weight(y):
    """
    Compute sample weight for unbalance labeling.

    Args:
        y (pandas.Serise or 1D array): labels

    Returns:
        pandas.Serise or 1D array: sample weights
    """
    #p = y.mean()
    #q = 1 - p
    #sp = 1 / p / 2
    #sq = 1 / q / 2
    #return y * sp + (1 - y) * sq
    return compute_sample_weight(class_weight="balanced", y=y)


class SelectionPipeline:
    """
    The basic pipeline for selection methods. It includes 2 parts: Scoring and Choosing.
    The detail methods is to be determinded.

    """

    def __init__(self):
        """
        Initialize the selection pipeline.

        Args:
        """
        self.name = "base"
        self.scores = None
        self.selected_score = None

    def Scoring(self, x, y=None):
        """
        The method to scores features is to be implemented.

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.

        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.
        """
        self.scores = x.max().sort_values(ascending=False)  # top is better
        return self.scores.copy()

    def Choose(self, scores, k):
        """
        Choosing features according to scores.

        Args:
            scores (pandas.Series or pandas.DataFrame): The score for each feature. Some elements may be empty.
            k (int): Number of feature to select. The result may less than k

        Returns:
            pandas.Series or pandas.DataFrame: The score for k selected features. May less than k.
        """
        self.selected_score = scores.head(k)
        self.selected_score = self.selected_score[self.selected_score != 0]

        return self.selected_score.copy()

    def Select(self, x, y, k):
        """
        A functional stack of: Scoring -> Choosing

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods.
            k (int): Number of feature to select. The result may less than k.

        Returns:
            pandas.Series: The score for k selected features. May less than k.
        """
        # x should be a pd dataframe or a numpy array without missing value
        scores = self.Scoring(x, y)
        selected_score = self.Choose(scores, k)
        return selected_score

    def Plotting(self):
        """
        plot hist graph of selectied feature importance
        """
        fig, ax = plt.subplots(1, 1)
        ax.bar(self.selected_score.index, self.selected_score)
        for label in ax.get_xticklabels(which='major'):
            label.set(rotation=45, horizontalalignment='right')
        ax.set_title(self.name + " score")

        plt.show()

    def Diagnose(self):
        """
        Give diagnose of selection.
        """
        #pass

    def Report(self):
        """
        Give diagnose of selection.
        """
        #pass
