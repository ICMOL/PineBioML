Module package.preprocessing.bagging
====================================

Classes
-------

`bagger()`
:   A wrapper of sparser_node

    ### Methods

    `fit(self, x, y=None)`
    :   Fit the string tree dictionary.
        Args:
            x (pandas.DataFrame): dataframe. should have .columns attribute
            y (pandas.Serise or 1D array, optional): label or target. This have no effects. Defaults to None.
        
        Returns:
            bagger: Fitted self.
        
        To do:
            transform

    `fit_transform(self, x, y=None)`
    :   A functional stack of fit and calling the result
        
        Args:
            x (_type_): _description_
            y (_type_, optional): _description_. Defaults to None.
        
        Returns:
            _type_: _description_

    `see(self, idx)`
    :   A functional interface of sparser_node.see
        
        Args:
            idx (string): see sparser_node.see
        
        Returns:
            node or list: see sparser_node.see

    `unbagging(self, name)`
    :   inverse transform of bagging
        
        Args:
            name (string): bagging name to query
        
        Returns:
            list: items in this bag.

`sparser_node(words, k)`
:   The kernel of string tree dictionary. It will sparse the feature name and then bagging similar feature with high corelation.
    
    For all feature names of a node in k-th layer, their first k digits will be the same.
    
    For all feature names of a node, they will be bagged together.
    
    Args:
        words (_type_): strings to sparse
        k (_type_): deepth of node.

    ### Methods

    `bagging(self, df)`
    :   Compute the explained variance ratio of 1st PC in PCA of all coresponding features in this node and its child.
        If the ratio > 0.9, the this node will be bagged. 
        The node name will be bagged feature name and 1st pc will be corespoding feature value.
        
        Args:
            df (_type_): The dataframe of all data.
        
        Returns:
            list: the list of all items in this node

    `collect(self, return_bag=True)`
    :   collect all items in its children and itself.
        
        Args:
            return_bag (bool, optional): True will return in bagged result, otherwise not. Defaults to True.
        
        Returns:
            list: the list of all items in this node and its children.

    `see(self, idx)`
    :   Query a specific node.
        
        if idx is empty, then the target is found and its item will be return
        else query its child by the first digit and pop the digit.
        
        Args:
            idx (string): node name
        
        Returns:
            list or node: if idx is empty, then the target is found and its item will be return. else query its child by the first digit and pop the digit.