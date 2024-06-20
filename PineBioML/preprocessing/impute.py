from sklearn.impute import KNNImputer, SimpleImputer
import pandas as pd
import numpy as np


class imputer():
    """
    To impute missing value. Include 4 parts:

    1. Deleting the features with too high missing value ratio.    
    2. Normalize input. Some imputed methods are sensitive to scale.    
    3. impute. method to be determinded.    
    4. inverse normalize.    
    """

    def __init__(self, threshold=0.333, center=True, scale=True):
        """__init__ _summary_

        Args:
            threshold (float): float from (0, 1]. If missing value rate of a feature is higher than threshold, it will be deleted. Defaults to 0.333
            center (bool, optional): _description_. Defaults to True.
            scale (bool, optional): _description_. Defaults to True.

        Raises:
            ValueError: missing value threshold must be a float in (0, 1]
        """
        # threshold sould between 0 ~ 1
        if 0 < threshold < 1 or threshold == 1:
            self.threshold = threshold
        else:
            raise ValueError(
                "missing value threshold must be a float in (0, 1]: ",
                threshold)
        self.normalizer = Normalizer(center=center, scale=scale)
        self.fitted = False

    def fit(self, x, y=None):
        """
        Fit x.
    
        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.

        Returns:
            inputer: fitted self
        """
        # drop too empty features
        self.not_too_empty = x.isna().mean() <= self.threshold
        x = x.loc[:, self.not_too_empty]  # keep those who not too empty

        # normalize
        x, y = self.normalizer.fit_transform(x, y)

        # call the kernel
        self.kernel.fit(x)
        self.fitted = True
        return self

    def transform(self, x, y=None):
        """
        Transform x

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.

        Returns:
            pandas.DataFrame or a 2D array: imputed x
        """
        if not self.fitted:
            raise "please call fit before calling transform."
        # drop too empty features
        x = x.loc[:, self.not_too_empty]  # keep those who not too empty

        columns = x.columns
        idx = x.index

        # normalize
        x, y = self.normalizer.fit_transform(x, y)

        # call the kernel
        x = self.kernel.transform(x)

        # rebuild the dataframe from numpy array returns
        x = pd.DataFrame(x, columns=columns, index=idx)

        # inverse normalize
        x, y = self.normalizer.inverse_transform(x, y)
        return x, y

    def fit_transform(self, x, y=None):
        """
        A functional stack of fit and transform.

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.

        Returns:
            pandas.DataFrame or a 2D array: imputed x
        """
        self.fit(x, y)
        x, y = self.transform(x, y)
        return x, y


class knn_imputer(imputer):
    """
    Using K nearest neighbor to impute missing value

    """

    def __init__(self, threshold=0.333, n_neighbor=5):
        """

        Args:
            threshold (float): float from (0, 1]. If missing value rate of a feature is higher than threshold, it will be deleted. Defaults to 0.333
            n_neighbor (int, optional): Number of nearest neighbor to use. Defaults to 5.
        """
        super().__init__(threshold)

        self.kernel = KNNImputer(n_neighbors=n_neighbor)


class simple_imputer(imputer):
    """
    Using a constant value (0, mean, median... etc.) to impute all missing value in a feature.
    """

    def __init__(self, threshold=0.3333, strategy="median"):
        """

        Args:
            threshold (float): float from (0, 1]. If missing value rate of a feature is higher than threshold, it will be deleted. Defaults to 0.333
            strategy (str, optional): The strategy to impute. One of {"mean", "median", "constant"}. Defaults to "median".
        """
        super().__init__(threshold, center=False, scale=False)

        self.kernel = SimpleImputer(strategy=strategy)


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

    def __init__(self, center=True, scale=True, global_scale=False):
        """
        Args:
            center (bool, optional): Whether to centralize in selection preprocessing. For input X, if ture then X = X- X.mean(axis = 0)    
            scale (bool, optional): Whether to scale after centralized. For input X, if ture then X = X / X.std(axis = 0). scale will overlap the effects of global_scale.
            global_scale (bool, optional): Whether to scale data in global. For input X, if ture then X = X / X.std(axis = [0, 1]). One of scale or global_scale sould be True, or lasso will raise an numerical error. 

        """
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
        if self.center:
            x = x - self.mean
        if self.scale:
            x = x / self.norm
        if self.global_scale:
            x = x / self.global_norm
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
        if self.global_scale:
            x = x * self.global_norm
        if self.scale:
            x = x * self.norm
        if self.center:
            x = x + self.mean
        return x, y
