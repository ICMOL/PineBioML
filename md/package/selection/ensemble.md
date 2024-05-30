Module package.selection.ensemble
=================================

Classes
-------

`selector(center=True, scale=True, log_domain=False)`
:   A functional stack of diffirent methods.
    
    Args:
        center (bool, optional): Pass to Normalizer. Defaults to True.
        scale (bool, optional): Pass to Normalizer. Defaults to True.
        log_domain (bool, optional): Whether input data is in log_domain. Defaults to False.

    ### Ancestors (in MRO)

    * package.selection.SelectionPipeline

    ### Methods

    `Select(self, x, y, k)`
    :   Calling all the methods in kernel sequancially.
        
        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.
            k (int): Number of feature to select. The result may less than k
        
        Returns:
            pandas.Series: The concatenated results. Top k (may less than k) important feature from diffient methods.

    `plotting(self)`
    :