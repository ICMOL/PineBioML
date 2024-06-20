import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def sample_weight(y):
    """
    Compute sample weight for unbalance labeling.

    Args:
        y (pandas.Serise or 1D array): labels

    Returns:
        pandas.Serise or 1D array: sample weights
    """
    p = y.mean()
    q = 1 - p
    sp = 1 / p / 2
    sq = 1 / q / 2
    return y * sp + (1 - y) * sq


class Normalizer:
    """ 
    A preprocessing class for selection methods.

    Defaults to standarization. For input X, it will sequantially do :    
        1. If center then X = X - mean(axis = 0)    
        2. If scale then X = X / X.std(axis = 0)    
        3. If global_scale then X = X / X.std(axis = [0, 1])    

    SVM-based and Lasso-based methods are sensitive to the the scale of input (in numerical and in result).    

    To do:
        The support of box-cox transform and power transform.
    """

    def __init__(self, center=True, scale=True, global_scale=False, axis=0):
        """
        Args:
            center (bool, optional): Whether to centralize in selection preprocessing. For input X, if ture then X = X- X.mean(axis = 0)    
            scale (bool, optional): Whether to scale after centralized. For input X, if ture then X = X / X.std(axis = 0). scale will overlap the effects of global_scale.
            global_scale (bool, optional): Whether to scale data in global. For input X, if ture then X = X / X.std(axis = [0, 1]). One of scale or global_scale sould be True, or lasso will raise an numerical error. 

        """
        self.axis = axis
        self.center = center
        self.mean = 0
        self.scale = scale
        self.norm = 1
        self.global_scale = global_scale
        self.global_norm = 1
        self.fitted = False

    def fit(self, x, y=None):
        """
        Computing and Recording the mean and std of X for prediction.    

        Args:
            x (pandas.DataFrame or a 2D array): The data to normalize.    
            y (pandas.Series or a 1D array): A placeholder only. Normalizer do nothing to y.    

        Returns:
            Normalizer: self after fitting.
        """
        if self.axis:
            x = x.T

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

        if self.axis:
            x = x.T
        self.fitted = True
        return self

    def transform(self, x, y=None):
        """
        Transform input x. Only activates after "fit" was called.

        Args:
            x (pandas.DataFrame or a 2D array): The data to normalize.    
            y (pandas.Series or a 1D array): A placeholder only. Normalizer do nothing to y.    

        Returns:
            pandas.DataFrame or a 2D array: Normalized x.
            pandas.Series or a 1D array: Same as input y.
        """
        if not self.fitted:
            print("WARNING: please call fit before calling transform")
        #if self.log_transform:
        #   x = np.log(x)
        if self.axis:
            x = x.T
        if self.center:
            x = x - self.mean
        if self.scale:
            x = x / self.norm
        if self.global_scale:
            x = x / self.global_norm
        if self.axis:
            x = x.T
        x_normalized = x
        return x_normalized, y

    def fit_transform(self, x, y=None):
        """
        A functional stack of "fit" and "transform".

        Args:
            x (pandas.DataFrame or a 2D array): The data to normalize.
            y (pandas.Series or a 1D array): A placeholder only. Normalizer do nothing to y.

        Returns:
            pandas.DataFrame or a 2D array: Normalized x.
        """
        self.fit(x, y)
        x_normalized = self.transform(x, y)
        return x_normalized

    def inverse_transform(self, x, y=None):
        """
        The inverse transform of normalize.

        Args:
            x (pandas.DataFrame or a 2D array): The data to revert original scale.
            y (pandas.Series or a 1D array): A placeholder only. Normalizer do nothing to y.

        Returns:
            pandas.DataFrame or a 2D array: x in original scale.
        """
        if self.axis:
            x = x.T
        if self.global_scale:
            x = x * self.global_norm
        if self.scale:
            x = x * self.norm
        if self.center:
            x = x + self.mean
        if self.axis:
            x = x.T
        return x, y


class SelectionPipeline:
    """
    The basic pipeline for selection methods. It includes 3 parts: Normalize, Scoring and Choosing.
    The detail methods is to be determinded.

    """

    def __init__(self, center=True, scale=True, global_scale=False):
        """
        Initialize the selection pipeline and Normalizer.

        Args:
            center (bool, optional): Pass to Normalizer. Defaults to True.
            scale (bool, optional): Pass to Normalizer. Defaults to True.
            global_scale (bool, optional): Pass to Normalizer. Defaults to False.
        """
        self.normalizer = Normalizer(center=center,
                                     scale=scale,
                                     global_scale=global_scale)
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
        A functional stack of: Normalizer -> Scoring -> Choosing

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods.
            k (int): Number of feature to select. The result may less than k.

        Returns:
            pandas.Series: The score for k selected features. May less than k.
        """
        # x should be a pd dataframe or a numpy array without missing value
        x, y = self.normalizer.fit_transform(x, y)
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
