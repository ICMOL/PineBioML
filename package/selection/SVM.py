import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from . import SelectionPipeline


class SVM_selection(SelectionPipeline):
    """
    Using the support vector of linear support vector classifier as scoring method.

    SVM_selection is scale sensitive in result.

    """

    def __init__(self, center=True, scale=True):
        """
        Args:
            center (bool, optional): _description_. Defaults to True.
            scale (bool, optional): _description_. Defaults to True.
        """
        super().__init__(center=center, scale=scale, global_scale=True)
        self.kernel = LinearSVC(dual="auto", class_weight="balanced")
        self.name = "SVM"

    def Scoring(self, x, y=None):
        """
        Using the support vector of linear support vector classifier as scoring method.

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.

        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.
        """
        self.kernel.fit(x, y)
        svm_weights = np.abs(self.kernel.coef_).sum(axis=0)
        svm_weights /= svm_weights.sum()

        self.scores = pd.Series(svm_weights, index=x.columns,
                                name=self.name).sort_values(ascending=False)
        return self.scores.copy()


"""
class multi_SVM_selection(SelectionPipeline):

    def __init__(self, center=True, scale=True):
        self.center = center
        self.scale = scale
        self.name = "multi_svm"

    def Select(self, x, y, k, batch_k=5):
        result = []
        if k == -1:
            k = x.shape[0]
        for i in range(k // batch_k):
            result.append(
                SVM_selection(center=self.center,
                              scale=self.scale).select(x, y, batch_k))
            x = x.drop(result[-1].index, axis=1)

        result = pd.concat(result)
        result = result - result.min()
        result.name = self.name

        self.selected_score = result.sort_values(ascending=False)
        return self.selected_score.copy()
"""
