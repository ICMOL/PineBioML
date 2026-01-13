import matplotlib.pyplot as plt
from pandas import concat
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold
# Suppress only ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class SelectionPipeline:
    """
    The basic pipeline for selection methods. It includes 2 parts: Scoring and Choosing.
    The detail methods is to be determinded.

    """

    def __init__(self,
                 k: int = None,
                 z_importance_threshold: float = 1.,
                 n_cv: int = 1):
        """
        Initialize the selection pipeline.

        Args:
            k (int or None): select top k important feature. k = -1 means selecting all, k = None means selecting the feature that have standarized score > 1. Default = None
            z_importance_threshold (int, optional): The threshold to picking features. Defaults to 1.
        """
        self.n_cv = n_cv
        self.k = k
        self.z_importance_threshold = z_importance_threshold
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
        scores = x.max().sort_values(ascending=False)  # top is better
        return scores

    def Select(self, scores):
        """
        pick features according to scores.
        if k == None, choose k such that:    
            z-scores = (scores - scores.mean())/scores.std()    
            k = # (z-scores > 1)
            
        Args:
            scores (pandas.Series): The score for each feature. Some elements may be empty.
        Returns:
            pandas.Series: The score for k selected features. May less than k.
        """
        # x should be a pd dataframe or a numpy array without missing value
        if self.k:
            # k not None
            selected_score = scores.head(self.k)
            selected_score = selected_score[selected_score != 0]
        else:
            # k is None
            z_scores = (scores - scores.mean()) / (scores.std() + 1e-4)
            selected_score = scores[z_scores > self.z_importance_threshold]
        selected_score.name = self.name
        return selected_score

    def fit(self, x, y):
        """
        sklearn api

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods.
        """
        if self.n_cv > 1:
            scores = []
            for train_idx, valid_idx in KFold(n_splits=self.n_cv,
                                              shuffle=True,
                                              random_state=20250915).split([
                                                  i for i in range(x.shape[0])
                                              ]):
                scores.append(
                    self.Scoring(x.iloc[train_idx], y.iloc[train_idx]))
            self.scores = (concat(scores, axis=1).sum(axis=1) /
                           self.n_cv).sort_values(ascending=False)
        else:
            self.scores = self.Scoring(x.copy(), y.copy())
        self.selected_score = self.Select(self.scores.copy())
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
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.size'] = 14
        fig, ax = plt.subplots(1, 1)
        ax.bar(self.selected_score.index, self.selected_score)
        for label in ax.get_xticklabels(which='major'):
            label.set(rotation=45, horizontalalignment='right')
        ax.set_title(self.name + " score")
        ax.set_ylabel("Importance")

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
