Module package.selection.SVM
============================

Classes
-------

`SVM_selection(center=True, scale=True)`
:   Using the support vector of linear support vector classifier as scoring method.
    
    SVM_selection is scale sensitive in result.
    
    Args:
        center (bool, optional): _description_. Defaults to True.
        scale (bool, optional): _description_. Defaults to True.

    ### Ancestors (in MRO)

    * package.selection.SelectionPipeline

    ### Methods

    `Scoring(self, x, y=None)`
    :   Using the support vector of linear support vector classifier as scoring method.
        
        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.
        
        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.