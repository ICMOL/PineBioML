import pandas as pd
from numpy import sqrt
from . import Normalizer


class simple_clip():

    def __init__(self, clip_quantile=0.95):
        """
        A simple clip method to remove outliers. It will clip each feature to their clip_quantile.

        Args:
            clip_quantile (float, optional): . Defaults to 0.95.
        """
        self.clip_quantile = clip_quantile
        self.fitted = False

    def fit(self, x: pd.DataFrame, y: pd.Series = None, sample_weight=None):
        """        
    
        Args:
            x (pandas.DataFrame or a 2D array): The data to compute quantile.
            y (pandas.Series or a 1D array): Null place holder, no effect. Defaults to None.

        Returns:
            self: fitted self
        """
        self.upper = x.quantile(self.clip_quantile)
        self.lower = x.quantile(1 - self.clip_quantile)
        self.fitted = True
        return self

    def transform(self, x: pd.DataFrame):
        """
        clip x to upper clip_quantile and lower clip_quantile.

        Args:
            x (pandas.DataFrame or a 2D array): The data to be transformed.

        Returns:
            pandas.DataFrame or a 2D array: transformed x
        """
        if not self.fitted:
            raise "please call fit before calling transform."

        return x.clip(self.lower, self.upper, axis=1)

    def fit_transform(self,
                      x: pd.DataFrame,
                      y: pd.Series = None,
                      sample_weight=None):
        """
        A functional stack of fit and transform.

        Args:
            x (pandas.DataFrame or a 2D array): The data to compute quantile.
            y (pandas.Series or a 1D array): Null place holder, no effect. Defaults to None.

        Returns:
            pandas.DataFrame or a 2D array: transformed x
        """
        self.fit(x, y)
        x = self.transform(x)
        return x


class IsolationForest():
    """
    A wrapper for IsolationForest from sklearn. It will remove outliers based on the IsolationForest algorithm.

    Args:

    """

    def __init__(self):
        self.dropped = []
        self.fitted = False

    def fit(self, x: pd.DataFrame, y: pd.Series = None, sample_weight=None):
        """Fit the IsolationForest model.

        Args:
            x (pandas.DataFrame or a 2D array): The data to fit the model.
            y (pandas.Series or a 1D array): Null place holder, no effect. Defaults to None.

        Returns:
            self: fitted self
        """
        from sklearn.ensemble import IsolationForest

        n_features = x.shape[1]
        self.model = IsolationForest(n_estimators=4 * round(sqrt(n_features)),
                                     max_features=round(sqrt(n_features)),
                                     n_jobs=-1,
                                     random_state=143,
                                     bootstrap=True,
                                     max_samples=0.7)
        self.model.fit(x)
        self.fitted = True
        return self

    def transform(self, x: pd.DataFrame):
        """
        Filter the data using the fitted IsolationForest model.

        Args:
            x (pandas.DataFrame or a 2D array): The data to transform.

        Returns:
            pandas.DataFrame or a 2D array: filterd x
        """
        if not self.fitted:
            raise "please call fit before calling transform."
        abnormal = self.model.predict(x)
        self.dropped.append(x.index[abnormal != 1])
        return x.loc[abnormal == 1]

    def fit_transform(self,
                      x: pd.DataFrame,
                      y: pd.Series = None,
                      sample_weight=None):
        """
        A functional stack of fit and transform.

        Args:
            x (pandas.DataFrame or a 2D array): The data to transform.
            y (pandas.Series or a 1D array): Null placeholder, no effect. Defaults to None.

        Returns:
            pandas.DataFrame or a 2D array: Filterd x
        """
        self.fit(x, y)
        x = self.transform(x)
        return x
