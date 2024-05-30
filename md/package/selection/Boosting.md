Module package.selection.Boosting
=================================

Classes
-------

`AdaBoost_selection(unbalanced=True, n_iter=128, learning_rate=0.01, center=True, scale=True)`
:   Using AdaBoost to scoring (gini impurity / entropy) features.
    
    Warning: If data is too easy, boosting methods is difficult to give score to all features.
    
    Args:
        n_iter (int, optional): Number of trees also number of iteration to boost. Defaults to 64.
        learning_rate (float, optional): boosting learning rate. Defaults to 0.01.
        unbalanced (bool, optional): True to imply class weight to samples. Defaults to False.
        center (bool, optional): Pass to Normalizer. Defaults to True.
        scale (bool, optional): Pass to Normalizer. Defaults to True.

    ### Ancestors (in MRO)

    * package.selection.SelectionPipeline

    ### Methods

    `Scoring(self, x, y=None)`
    :   Using AdaBoost to scoring (gini impurity / entropy) features.
        
        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.
        
        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.

`Lightgbm_selection(unbalanced=True, center=True, scale=False)`
:   Using Lightgbm to scoring (gini impurity / entropy) features. 
    
    Warning: If data is too easy, boosting methods is difficult to give score to all features.
    
    Args:
        unbalanced (bool, optional): True to imply class weight to samples. Defaults to False.
        center (bool, optional): Pass to Normalizer. Defaults to True.
        scale (bool, optional): Pass to Normalizer. Defaults to True.

    ### Ancestors (in MRO)

    * package.selection.SelectionPipeline

    ### Methods

    `Scoring(self, x, y=None)`
    :   Using Lightgbm to scoring (gini impurity / entropy) features.
        
        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.
        
        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.

`XGboost_selection(unbalanced=True, center=True, scale=True)`
:   Using XGboost to scoring (gini impurity / entropy) features.
    
    Warning: If data is too easy, boosting methods is difficult to give score to all features.
    
    Args:
        unbalanced (bool, optional): True to imply class weight to samples. Defaults to False.
        center (bool, optional): Pass to Normalizer. Defaults to True.
        scale (bool, optional): Pass to Normalizer. Defaults to True.

    ### Ancestors (in MRO)

    * package.selection.SelectionPipeline

    ### Methods

    `Scoring(self, x, y=None)`
    :   Using XGboost to scoring (gini impurity / entropy) features.
        
        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.
        
        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.