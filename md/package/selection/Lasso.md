Module package.selection.Lasso
==============================

Classes
-------

`Lasso_bisection_selection(center=True, scale=True, unbalanced=True, objective='BinaryClassificaation')`
:   Using Lasso (L1 penalty) regression as scoring method.  More specifically, L1 penalty will force feature weights to be zeros. 
    As the coefficient of penalty increases, more and more weights of features got killed and the important feature will remain.
    
    Lasso_bisection_selection will use binary search to find out when all weights vanish.
    
    The trace of weight vanishment is not support.
    
    Args:
        unbalanced (bool, optional): False to imply class weight to samples. Defaults to True.
        center (bool, optional): Pass to Normalizer. Defaults to True.
        scale (bool, optional): Pass to Normalizer. Defaults to True.
        objective (str, optional): one of {"Regression", "BinaryClassification"}

    ### Ancestors (in MRO)

    * package.selection.SelectionPipeline

    ### Methods

    `Select(self, x, y, k)`
    :   Using Lasso (L1 penalty) regression as scoring method.  More specifically, L1 penalty will force feature weights to be zeros. 
        As the coefficient of penalty increases, more and more weights of features got killed and the important feature will remain.
        
        Lasso_bisection_selection will use binary search to find out when all weights vanish.
        
        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.
            k (int): Number of feature to select. The result may less than k
        
        Returns:
            pandas.Series: The score for k selected features. May less than k.

    `create_kernel(self, C)`
    :

`Lasso_selection(unbalanced=True, center=True, scale=True, objective='BinaryClassificaation')`
:   Using Lasso (L1 penalty) regression as scoring method.  More specifically, L1 penalty will force feature weights to be zeros. 
    As the coefficient of penalty increases, more and more weights of features got killed and the important feature will remain.
    
    Lasso_selection will use grid search to find out when all weights vanish.
    
    Lasso_selection is scale sensitive in numerical and in result.
    
    Args:
        unbalanced (bool, optional): False to imply class weight to samples. Defaults to True.
        center (bool, optional): Pass to Normalizer. Defaults to True.
        scale (bool, optional): Pass to Normalizer. Defaults to True.
        objective (str, optional): one of {"Regression", "BinaryClassification"}

    ### Ancestors (in MRO)

    * package.selection.SelectionPipeline

    ### Methods

    `Scoring(self, x, y=None)`
    :   Using Lasso (L1 penalty) regression as scoring method.  More specifically, L1 penalty will force feature weights to be zeros. 
        As the coefficient of penalty increases, more and more weights of features got killed and the important feature will remain.
        
        Lasso_selection will use grid search to find out when all weights vanish.
        
         Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.
        
        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.
        
        To do:
            kfold validation performance threshold.

    `create_kernel(self, C)`
    :   Create diffirent kernel according to opjective.
        
        Args:
            C (float): The coefficient to L1 penalty.
        
        Returns:
            sklearn.linearmodel: a kernel of sklearn linearmodel

`multi_Lasso_selection(center=True, scale=True, objective='BinaryClassificaation')`
:   A stack of Lasso_bisection_selection. Because of collinearity, if there are a batch of featres with high corelation, only one of them will remain.
    That leads to diffirent behavior between select k features in a time and select k//n features in n times.
    
    Args:
        center (bool, optional): Pass to Normalizer. Defaults to True.
        scale (bool, optional): Pass to Normalizer. Defaults to True.
        objective (str, optional): one of {"Regression", "BinaryClassification"}

    ### Ancestors (in MRO)

    * package.selection.SelectionPipeline

    ### Methods

    `Select(self, x, y, k, n=5)`
    :   Select k//n features for n times, and then concatenate the results.
        
        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.
            k (int): Number of feature to select. The result may less than k
            n (int, optional): Number of batch which splits k to select. Defaults to 10.
        
        Returns:
            pandas.Series: The score for k selected features. May less than k.