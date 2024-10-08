from . import SelectionPipeline
from . import sample_weight as compute_sample_weight
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgbm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import OneHotEncoder


class XGboost_selection(SelectionPipeline):
    """
    Using XGboost to scoring (gini impurity / entropy) features.

    Warning: If data is too easy, boosting methods is difficult to give score to all features.
    """

    def __init__(self, k, unbalanced=True):
        """
        Args:
            unbalanced (bool, optional): True to imply class weight to samples. Defaults to False.
        """
        super().__init__(k=k)
        self.unbalanced = unbalanced

        self.kernel = xgb.XGBClassifier(random_state=142, subsample=0.7)
        self.name = "XGboost"

    def Scoring(self, x, y=None):
        """
        Using XGboost to scoring (gini impurity / entropy) features.

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.

        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.
        """
        if self.unbalanced:
            sample_weight = compute_sample_weight(y)
        else:
            sample_weight = np.ones(len(y))

        y = OneHotEncoder(sparse_output=False).fit_transform(
            y.to_numpy().reshape(-1, 1))
        self.kernel.fit(x, y, sample_weight=sample_weight)
        score = self.kernel.feature_importances_
        self.scores = pd.Series(score, index=x.columns,
                                name=self.name).sort_values(ascending=False)
        return self.scores.copy()


class Lightgbm_selection(SelectionPipeline):
    """
    Using Lightgbm to scoring (gini impurity / entropy) features. 

    Warning: If data is too easy, boosting methods is difficult to give score to all features.
    """

    def __init__(self, k, unbalanced=True):
        """
        Args:
            unbalanced (bool, optional): True to imply class weight to samples. Defaults to False.
        """
        super().__init__(k=k)
        self.unbalanced = unbalanced

        self.kernel = lgbm.LGBMClassifier(learning_rate=0.01,
                                          random_state=142,
                                          subsample=0.7,
                                          subsample_freq=1)
        self.name = "Lightgbm"

    def Scoring(self, x, y=None):
        """
        Using Lightgbm to scoring (gini impurity / entropy) features.

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.

        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.
        """
        if self.unbalanced:
            sample_weight = compute_sample_weight(y)
        else:
            sample_weight = np.ones(len(y))

        self.kernel.fit(x, y, sample_weight=sample_weight)
        score = self.kernel.feature_importances_
        self.scores = pd.Series(score, index=x.columns,
                                name=self.name).sort_values(ascending=False)
        return self.scores.copy()


class AdaBoost_selection(SelectionPipeline):
    """
    Using AdaBoost to scoring (gini impurity / entropy) features.

    Warning: If data is too easy, boosting methods is difficult to give score to all features.
    """

    def __init__(self, k, unbalanced=True, n_iter=128, learning_rate=0.01):
        """
        Args:
            n_iter (int, optional): Number of trees also number of iteration to boost. Defaults to 64.
            learning_rate (float, optional): boosting learning rate. Defaults to 0.01.
            unbalanced (bool, optional): True to imply class weight to samples. Defaults to False.
        """
        super().__init__(k=k)
        self.unbalanced = unbalanced
        self.kernel = AdaBoostClassifier(
            n_estimators=n_iter,
            learning_rate=learning_rate,
            random_state=142,
        )
        self.name = "AdaBoost" + str(n_iter)

    def Scoring(self, x, y=None):
        """
        Using AdaBoost to scoring (gini impurity / entropy) features.

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.

        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.
        """
        print("I don't have a progress bar but I am running")
        if self.unbalanced:
            sample_weight = compute_sample_weight(y)
        else:
            sample_weight = np.ones(len(y))
        self.kernel.fit(x, y, sample_weight)
        score = self.kernel.feature_importances_
        self.scores = pd.Series(score, index=x.columns,
                                name=self.name).sort_values(ascending=False)
        return self.scores.copy()
