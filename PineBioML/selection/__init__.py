import matplotlib.pyplot as plt

import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress only ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class SelectionPipeline:
    """
    The basic pipeline for selection methods. It includes 2 parts: Scoring and Choosing.
    The detail methods is to be determinded.

    """

    def __init__(self, k=None):
        """
        Initialize the selection pipeline.

        Args:
            k (int or None): select top k important feature. k = -1 means selecting all, k = None means selecting the feature that have standarized score > 1. Default = None
        """
        self.k = k
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

    def Choose(self, scores):
        """
        Choosing features according to scores.

        Args:
            scores (pandas.Series or pandas.DataFrame): The score for each feature. Some elements may be empty.
            k (int): Number of feature to select. The result may less than k

        Returns:
            pandas.Series or pandas.DataFrame: The score for k selected features. May less than k.
        """
        self.selected_score = scores.head(self.k)
        self.selected_score = self.selected_score[self.selected_score != 0]

        return self.selected_score.copy()

    def Select(self, x, y):
        """
        A functional stack of: Scoring and Choosing    
        if k == None, choose k such that:    
            z-scores = (scores - scores.mean())/scores.std()    
            k = # (z-scores > 1)
            

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods.
            k (int): Number of feature to select. The result may less than k.

        Returns:
            pandas.Series: The score for k selected features. May less than k.
        """
        # x should be a pd dataframe or a numpy array without missing value
        scores = self.Scoring(x, y)
        if self.k:
            # k not None
            selected_score = self.Choose(scores)
        else:
            # k is None
            z_scores = (scores - scores.mean()) / (scores.std() + 1e-4)
            selected_score = scores[z_scores > 1.]

        return selected_score

    def fit(self, x, y):
        """
        sklearn api

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods.
        """
        self.Select(x, y)
        return self

    def transform(self, x):
        return x[self.selected_score.index]

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)

    def what_matters(self):
        return self.selected_score

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

    def reference(self) -> dict[str, str]:
        """
        This function will return reference of this method in python dict.    
        If you want to access it in PineBioML api document, then click on the    >Expand source code     

        Returns:
            dict[str, str]: a dict of reference.
        """
        refers = {
            "sklearn publication":
            "https://dl.acm.org/doi/10.5555/1953048.2078195"
        }

        return refers

    def Report(self):
        """
        Give diagnose of selection.
        """
        #pass
