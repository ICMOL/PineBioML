Module package.selection.RF
===========================

Classes
-------

`RF_selection(trees=16384, unbalanced=True, strategy='gini', center=True, scale=True)`
:   Using random forest to scoring (gini impurity / entropy) features.
    
    Args:
        trees (int, optional): Number of trees. Defaults to 1024*16.
        strategy (str, optional): Scoring strategy, one of {"gini", "entropy"}. Defaults to "gini".
        unbalanced (bool, optional): True to imply class weight to samples. Defaults to True.
        center (bool, optional): Pass to Normalizer. Defaults to True.
        scale (bool, optional): Pass to Normalizer. Defaults to True.

    ### Ancestors (in MRO)

    * package.selection.SelectionPipeline

    ### Methods

    `Scoring(self, x, y=None)`
    :   Using random forest to scoring (gini impurity / entropy) features.
        
        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.
        
        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.

`oob_RFClassifier(trees=8192, unbalanced=True)`
:   A random forest implement with out-of-bag evaluation. Boostrap subsampling strategy using Bernoulli sampling.
    
    Args:
        trees (int, optional): Number of trees. Defaults to 1024*16.
        unbalanced (bool, optional): True to imply class weight to samples. Defaults to True.
    
    To do:
        Now is only for classification.

    ### Methods

    `evaluate(self, x, y, metric='ACC', oob=False)`
    :   Evaluate model using input x, y
        
        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods.
            metric (str, optional): One of {acc, f1, bce, auc}. Defaults to "ACC".
            oob (bool, optional): True to use out of bag evaluate. Defaults to False.
        
        Returns:
            float: Evaluation result

    `fit(self, x, y)`
    :   Training the random forest.
        
        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods.

    `oob_predict_prob(self, x)`
    :   x must be the training data.
        
        Give a ratio that how many trees from forest predict out-of-bag x as positive.
        
        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
        
        Returns:
            pandas.Series or a 1D array: ratio that how many trees from forest predict out-of-bag x as positive. Defaults to None.

    `predict_prob(self, x)`
    :   Give a ratio that how many trees in the forest predict x as positive.
        
        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
        
        Returns:
            pandas.Series or a 1D array: ratio that how many trees in the forest predict x as positive. Defaults to None.

`pcRF_selection(trees=512, unbalanced=True, strategy='permutation', factorize_method='PCA', center=True, scale=True)`
:   Expiriment method. PCA->RF->importance->inverse_PCA
    
    Args:
        trees (int, optional): Number of trees. Defaults to 512.
        unbalanced (bool, optional): _description_. Defaults to True.
        strategy (str, optional): Scoring strategy, one of {"gini", "entropy", "permutation"}. Defaults to "permutation".
        factorize_method (str, optional): One of {"PCA"}. Method to reduce dimension.  Defaults to "PCA".
        center (bool, optional): Pass to Normalizer. Defaults to True.
        scale (bool, optional): Pass to Normalizer. Defaults to True.

    ### Ancestors (in MRO)

    * package.selection.SelectionPipeline

    ### Methods

    `Scoring(self, x, y=None)`
    :   Using random forest to scoring (gini impurity / entropy) principal components.
        
        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.
        
        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.

    `Select(self, x, y, k)`
    :   x->PCA->RF + oob +permutation importance -> inverse PCA -> feature importance
        
        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.
            k (int): Number of feature to select. The result may less than k
        
        Returns:
            pandas.Series: The score for k selected features. May less than k.