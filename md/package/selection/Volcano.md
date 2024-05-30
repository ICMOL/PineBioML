Module package.selection.Volcano
================================

Classes
-------

`Volcano_selection(strategy='fold', p_threshold=0.05, fc_threshold=2, log_domain=False, center=True, scale=False, absolute=True)`
:   volcano plot.
    
    Args:
        strategy (str, optional): Choosing strategy. One of {"p" or "fold"} Defaults to "fold".
        p_threshold (float, optional): p-value threshold. Only feature has p-value higher than threshold will be considered. Defaults to 0.05.
        fc_threshold (int, optional): fold change threshold. Only feature has fold change higher than threshold will be considered. Defaults to 2.
        log_domain (bool, optional): Whether input data is in log_domain. Defaults to False.
        center (bool, optional): Pass to Normalizer. Defaults to True.
        scale (bool, optional): Pass to Normalizer. Defaults to True.
        absolute (bool, optional): If true, then take absolute value on score while strategy == "p". Defaults to True.

    ### Ancestors (in MRO)

    * package.selection.SelectionPipeline

    ### Methods

    `Choose(self, scores, k)`
    :   Choosing the features which has score higher than threshold in assigned strategy.
        
        If strategy == "fold": sort in fold change and return p-value
        
        If strategy == "p": sort in p-value and return fold change
        
        Args:
            scores (pandas.DataFrame): A dataframe records p-value and fold change.
            k (int): Number of features to select.
        
        Returns:
            pandas.Series: The score for k selected features in assigned strategy.

    `Scoring(self, x, y)`
    :   Compute the fold change and p-value on each feature.
        
        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.
        
        Returns:
           pandas.DataFrame: A dataframe records p-value and fold change.

    `plotting(self, external=False, external_score=None, title='Welch t-test volcano', show=True, saving=False, save_path='./output/')`
    :   Plotting
        
        Args:
            external (bool, optional): True to use external score. Defaults to False.
            external_score (_type_, optional): External score to be used. Only activate when external == True. Defaults to None.
            title (str, optional): plot title. Defaults to "Welch t-test volcano".
            show (bool, optional): True to show the plot. Defaults to True.
            saving (bool, optional): True to save the plot. Defaults to False.
            save_path (str, optional): The path to save plot. Only activate when saving == True. Defaults to "./output/images/".