Module package.selection.DT
===========================

Classes
-------

`DT_selection(bins=10, q=0.05, strategy='c45', center=True, scale=True)`
:   A child class of SelectionPipeline .
    
    Using Decision stump (a single Decision tree) to scoring features.
    
    Args:
        bins (int, optional): Bins to esimate data distribution entropy. Defaults to 10.
        q (float, optional): Clip data values out of [q, 1-q] percentile to reduce the affect of outliers while estimate entropy. Defaults to 0.05.
        strategy (str, optional): One of {"id3", "c45"}. The strategy to build decision tree. Defaults to "c45".
        center (bool, optional): Pass to Normalizer. Defaults to True.
        scale (bool, optional): Pass to Normalizer. Defaults to True.

    ### Ancestors (in MRO)

    * package.selection.SelectionPipeline

    ### Methods

    `Scoring(self, x, y=None)`
    :   Using Decision stump (a single Decision tree) to scoring features. Though, single layer stump is equivalent to compare the id3/c4.5 score directly.
        
        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.
        
        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.

    `entropy(self, x)`
    :   Estimate entropy
        
        Args:
            x (pandas.DataFrame): data with bined label
        
        Returns:
            float: entropy