Module package.selection
========================

Sub-modules
-----------
* package.selection.Boosting
* package.selection.DT
* package.selection.Lasso
* package.selection.RF
* package.selection.SVM
* package.selection.Volcano
* package.selection.ensemble

Functions
---------

    
`sample_weight(y)`
:   Compute sample weight for unbalance labeling.
    
    Args:
        y (pandas.Serise or 1D array): labels
    
    Returns:
        pandas.Serise or 1D array: sample weights

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

`SelectionPipeline(center=True, scale=True, global_scale=False)`
:   The basic pipeline for selection methods. It includes 3 parts: Normalize, Scoring and Choosing.
    The detail methods is to be determinded.
    
    Initialize the selection pipeline and Normalizer.
    
    Args:
        center (bool, optional): Pass to Normalizer. Defaults to True.
        scale (bool, optional): Pass to Normalizer. Defaults to True.
        global_scale (bool, optional): Pass to Normalizer. Defaults to False.

    ### Descendants

    * package.selection.Boosting.AdaBoost_selection
    * package.selection.Boosting.Lightgbm_selection
    * package.selection.Boosting.XGboost_selection
    * package.selection.DT.DT_selection
    * package.selection.Lasso.Lasso_bisection_selection
    * package.selection.Lasso.Lasso_selection
    * package.selection.Lasso.multi_Lasso_selection
    * package.selection.RF.RF_selection
    * package.selection.RF.pcRF_selection
    * package.selection.SVM.SVM_selection
    * package.selection.Volcano.Volcano_selection
    * package.selection.ensemble.selector

    ### Methods

    `Choose(self, scores, k)`
    :   Choosing features according to scores.
        
        Args:
            scores (pandas.Series or pandas.DataFrame): The score for each feature. Some elements may be empty.
            k (int): Number of feature to select. The result may less than k
        
        Returns:
            pandas.Series or pandas.DataFrame: The score for k selected features. May less than k.

    `Diagnose(self)`
    :   Give diagnose of selection.

    `Plotting(self)`
    :   plot hist graph of selectied feature importance

    `Report(self)`
    :   Give diagnose of selection.

    `Scoring(self, x, y=None)`
    :   The method to scores features is to be implemented.
        
        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.
        
        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.

    `Select(self, x, y, k)`
    :   A functional stack of: Normalizer -> Scoring -> Choosing
        
        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods.
            k (int): Number of feature to select. The result may less than k.
        
        Returns:
            pandas.Series: The score for k selected features. May less than k.