import sklearn.preprocessing as skprpr
from pandas import DataFrame
from sklearn.base import BaseEstimator


class RemoveDummy(BaseEstimator):
    """ 
    Remove dummy features. Dummy features are those with a constant value.
    """

    def __init__(self):
        pass

    def fit(self, x, y=None):
        """
        Finding the dummy features.

        Args:
            x (pandas.DataFrame or a 2D array): The data to normalize.
            y (pandas.Series or a 1D array): A placeholder only. Normalizer do nothing to y.

        Returns:
            remove: self after fitting.
        """

        self.to_drop_ = x.columns[x.nunique() <= 1]
        return self

    def transform(self, x):
        """
        Drop the dummy features. Only activates after "fit" was called.

        Args:
            x (pandas.DataFrame or a 2D array): The data to drop dummy features.

        Returns:
            pandas.DataFrame or a 2D array: Cleared x.
        """
        x_dropped = x.drop(self.to_drop_, axis=1)

        return x_dropped

    def fit_transform(self, x, y=None):
        """
        A functional stack of "fit" and "transform".

        Args:
            x (pandas.DataFrame or a 2D array): The data to normalize.
            y (pandas.Series or a 1D array): A placeholder only. Normalizer do nothing to y.

        Returns:
            pandas.DataFrame or a 2D array: Cleared x.
        """
        self.fit(x, y)
        x_dropped = self.transform(x)
        return x_dropped


class Normalizer(BaseEstimator):
    """ 
    A wrapper of sklearn normalizers. This will conserve pandas features.    
    method be one of ["StandardScaler", "RobustScaler", "MinMaxScaler", "Normalizer", "PowerTransformer"]    
    let x with shape [n, d] where n is the sample size, d is the number of features.    
        StandardScaler(x) = (x - x.mean(n)) / x.std(n)    
        RobustScaler(x) = (x - x.median(n)) / (x.quartile3(n) - x.quartile1(n))    
        MinMaxScaler(x) = (x - x.min(n)) / (x.max(n) - x.min(n))    
        Normalizer(x) = x / x.norm(d)
        PowerTransformer(x) = yeo-johnson transform(x)
    """

    def __init__(self, center=True, scale=True, method="RobustScaler"):
        """
        Args:
            center (bool, optional): Whether to centralize data. Default to True.
            scale (bool, optional): Whether to scaling data after centralized. Default to True.
            method (str, optional): the way to normalize data. Be one of ["StandardScaler", "RobustScaler", "MinMaxScaler", "Normalizer", "PowerTransformer"]
        """
        self.center = center
        self.scale = scale
        self.method = method

    def _set_up(self):
        kernels = {
            "StandardScaler":
            skprpr.StandardScaler(with_mean=self.center, with_std=self.scale),
            "RobustScaler":
            skprpr.RobustScaler(with_centering=self.center,
                                with_scaling=self.scale),
            "MinMaxScaler":
            skprpr.MinMaxScaler(),
            "Normalizer":
            skprpr.Normalizer(),
            "PowerTransformer":
            skprpr.PowerTransformer()
        }
        self.kernel_ = kernels[self.method]

    def fit(self, x, y=None):
        """
        Computing and Recording the mean and std of X for prediction.    

        Args:
            x (pandas.DataFrame or a 2D array): The data to normalize.    
            y (pandas.Series or a 1D array): A placeholder only. Normalizer do nothing to y.    

        Returns:
            Normalizer: self after fitting.
        """
        self._set_up()
        self.kernel_.fit(x, y)

        return self

    def transform(self, x):
        """
        Transform input x. Only activates after "fit" was called.

        Args:
            x (pandas.DataFrame or a 2D array): The data to normalize.    
            y (pandas.Series or a 1D array): A placeholder only. Normalizer do nothing to y.    

        Returns:
            pandas.DataFrame or a 2D array: Normalized x.
            pandas.Series or a 1D array: Same as input y.
        """
        x_normalized = DataFrame(self.kernel_.transform(x),
                                 index=x.index,
                                 columns=x.columns)

        return x_normalized

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
        x_normalized = self.transform(x)
        return x_normalized

    def inverse_transform(self, x):
        """
        The inverse transform of normalize.

        Args:
            x (pandas.DataFrame or a 2D array): The data to revert original scale.
            y (pandas.Series or a 1D array): A placeholder only. Normalizer do nothing to y.

        Returns:
            pandas.DataFrame or a 2D array: x in original scale.
        """

        x_original = DataFrame(self.kernel_.inverse_transform(x),
                               index=x.index,
                               columns=x.columns)

        return x_original


class Pass(BaseEstimator):
    """ 
    Do nothing.
    """

    def __init__(self):
        pass

    def fit(self, x, y=None):
        """
        Do nothing.    

        Args:
            x (pandas.DataFrame or a 2D array): The data to normalize.    
            y (pandas.Series or a 1D array): A placeholder only. Normalizer do nothing to y.    

        Returns:
            Pass: No, do nothing.
        """
        self.do_nothing_ = True

        return self

    def transform(self, x):
        """
        Do nothing.

        Args:
            x (pandas.DataFrame or a 2D array): The data to normalize.    

        Returns:
            pandas.DataFrame or a 2D array: Normalized x.
        """

        return x

    def fit_transform(self, x, y=None):
        """
        Do nothing.

        Args:
            x (pandas.DataFrame or a 2D array): The data to normalize.
            y (pandas.Series or a 1D array): A placeholder only. Normalizer do nothing to y.

        Returns:
            pandas.DataFrame or a 2D array: x.
        """
        self.fit(x, y)
        x_normalized = self.transform(x)
        return x_normalized

    def inverse_transform(self, x):
        """
        Do nothing.

        Args:
            x (pandas.DataFrame or a 2D array): The data to revert original scale.

        Returns:
            pandas.DataFrame or a 2D array: x.
        """

        return x
