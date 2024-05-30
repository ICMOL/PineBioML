Module package.preprocessing.impute
===================================

Classes
-------

`Normalizer(center=True, scale=True, global_scale=False)`
:   A preprocessing class for selection methods.
    
    Defaults to standarization. For input X, it will sequantially do :    
        1. If center then X = X - mean(axis = 0)    
        2. If scale then X = X / X.std(axis = 0)    
        3. If global_scale then X = X / X.std(axis = [0, 1])    
    
    SVM-based and Lasso-based methods are sensitive to the the scale of input (in numerical and in result).    
    
    To do:
        The support of box-cox transform and power transform.
    
    Args:
        center (bool, optional): Whether to centralize in selection preprocessing. For input X, if ture then X = X- X.mean(axis = 0)    
        scale (bool, optional): Whether to scale after centralized. For input X, if ture then X = X / X.std(axis = 0). scale will overlap the effects of global_scale.
        global_scale (bool, optional): Whether to scale data in global. For input X, if ture then X = X / X.std(axis = [0, 1]). One of scale or global_scale sould be True, or lasso will raise an numerical error.

    ### Methods

    `fit(self, x, y=None)`
    :   Computing and Recording the mean and std of X for prediction.    
        
        Args:
            x (pandas.DataFrame or a 2D array): The data to normalize.    
            y (pandas.Series or a 1D array): A placeholder only. Normalizer do nothing to y.    
        
        Returns:
            Normalizer: self after fitting.

    `fit_transform(self, x, y=None)`
    :   A functional stack of "fit" and "transform".
        
        Args:
            x (pandas.DataFrame or a 2D array): The data to normalize.
            y (pandas.Series or a 1D array): A placeholder only. Normalizer do nothing to y.
        
        Returns:
            pandas.DataFrame or a 2D array: Normalized x.

    `inverse_transform(self, x, y=None)`
    :   The inverse transform of normalize.
        
        Args:
            x (pandas.DataFrame or a 2D array): The data to revert original scale.
            y (pandas.Series or a 1D array): A placeholder only. Normalizer do nothing to y.
        
        Returns:
            pandas.DataFrame or a 2D array: x in original scale.

    `transform(self, x, y=None)`
    :   Transform input x. Only activates after "fit" was called.
        
        Args:
            x (pandas.DataFrame or a 2D array): The data to normalize.    
            y (pandas.Series or a 1D array): A placeholder only. Normalizer do nothing to y.    
        
        Returns:
            pandas.DataFrame or a 2D array: Normalized x.
            pandas.Series or a 1D array: Same as input y.

`imputer(threshold=0.333, center=True, scale=True)`
:   To impute missing value. Include 4 parts:
    
    1. Deleting the features with too high missing value ratio.    
    2. Normalize input. Some imputed methods are sensitive to scale.    
    3. impute. method to be determinded.    
    4. inverse normalize.    
    
    __init__ _summary_
    
    Args:
        threshold (float): float from (0, 1]. If missing value rate of a feature is higher than threshold, it will be deleted. Defaults to 0.333
        center (bool, optional): _description_. Defaults to True.
        scale (bool, optional): _description_. Defaults to True.
    
    Raises:
        ValueError: missing value threshold must be a float in (0, 1]

    ### Descendants

    * package.preprocessing.impute.knn_imputer
    * package.preprocessing.impute.simple_imputer

    ### Methods

    `fit(self, x, y=None)`
    :   Fit x.
        
        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.
        
        Returns:
            inputer: fitted self

    `fit_transform(self, x, y=None)`
    :   A functional stack of fit and transform.
        
        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.
        
        Returns:
            pandas.DataFrame or a 2D array: imputed x

    `transform(self, x, y=None)`
    :   Transform x
        
        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.
        
        Returns:
            pandas.DataFrame or a 2D array: imputed x

`knn_imputer(threshold=0.333, n_neighbor=5)`
:   Using K nearest neighbor to impute missing value
    
    Args:
        threshold (float): float from (0, 1]. If missing value rate of a feature is higher than threshold, it will be deleted. Defaults to 0.333
        n_neighbor (int, optional): Number of nearest neighbor to use. Defaults to 5.

    ### Ancestors (in MRO)

    * package.preprocessing.impute.imputer

`simple_imputer(threshold=0.3333, strategy='median')`
:   Using a constant value (0, mean, median... etc.) to impute all missing value in a feature.
    
    Args:
        threshold (float): float from (0, 1]. If missing value rate of a feature is higher than threshold, it will be deleted. Defaults to 0.333
        strategy (str, optional): The strategy to impute. One of {"mean", "median", "constant"}. Defaults to "median".

    ### Ancestors (in MRO)

    * package.preprocessing.impute.imputer