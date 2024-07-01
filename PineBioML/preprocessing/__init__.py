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
