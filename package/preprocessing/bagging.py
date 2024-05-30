from sklearn.decomposition import PCA

MAX_ITEM = 32
BAGGING_THRESHOLD = 0.9


class sparser_node:
    """ 
    The kernel of string tree dictionary. It will sparse the feature name and then bagging similar feature with high corelation.
    
    For all feature names of a node in k-th layer, their first k digits will be the same.

    For all feature names of a node, they will be bagged together.

    """

    def __init__(self, words, k):
        """
        Args:
            words (_type_): strings to sparse
            k (_type_): deepth of node.
        """
        self.k = k  # the depth of tree, also the digit of current sparser
        self.name = words[0][:self.k] + "*"  # name of this node
        self.bagged = False  # if the node is a bag, determined after call bagging function

        items = []
        collector = {}
        # separate the words which is not long enough
        # for example, got words = ["a", "b", "a1", "a2"]
        # then the items of this bag should be ["a", "b"]
        # the collector of child will be ["a1", "a2"]
        for word in words:
            if len(word) > self.k + 1:
                if word[self.k] not in collector:
                    collector[word[self.k]] = []
                collector[word[self.k]].append(word)
            else:
                items.append(word)

        self.items = items
        self.children = {}
        # ["a1", "a2"] => {"a": [a1, a2]}
        for symbol in collector:
            self.children[symbol] = sparser_node(collector[symbol], self.k + 1)

    def __call__(self, idx):
        """
        query the child.

        Args:
            idx (char): char to query

        Returns:
            sparse_node: Child coresponding to idx
        """
        return self.children[idx]

    def bagging(self, df):
        """
        Compute the explained variance ratio of 1st PC in PCA of all coresponding features in this node and its child.
        If the ratio > 0.9, the this node will be bagged. 
        The node name will be bagged feature name and 1st pc will be corespoding feature value.

        Args:
            df (_type_): The dataframe of all data.

        Returns:
            list: the list of all items in this node
        """
        items = []
        items += self.items

        for child in self.children:
            items += self.children[child].bagging(
                df)  # recurrently call bagging (deep first, digit tree)

        n = len(items)  # only 1 item => no bagging
        if n == 1:
            return items
        elif n < MAX_ITEM:  # max number of items for each bag
            data = df[items]
            ### bagging rules
            # rule 1
            #score = np.mean(data.corr().abs())
            # rule 2
            pca = PCA(n)
            tmp_data = pca.fit_transform((data - data.mean() / data.std()))
            score = pca.explained_variance_ratio_[0]

            if score > BAGGING_THRESHOLD:
                #print(np.mean(data.corr().abs()), pca.explained_variance_ratio_, "#################")
                #df[self.name] = df[items].mean(axis = 1) # act 1
                df[self.name] = tmp_data[:, 0]  # act 1

                self.bagged = True  # record this node is bagged
                return [self.name]  # bag name
        return items

    def collect(self, return_bag=True):
        """
        collect all items in its children and itself.

        Args:
            return_bag (bool, optional): True will return in bagged result, otherwise not. Defaults to True.

        Returns:
            list: the list of all items in this node and its children.
        """
        if self.bagged and return_bag:
            return [self.name]

        items = []
        items += self.items

        for child in self.children:
            items += self.children[child].collect(return_bag)
        return items

    def see(self, idx):
        """
        Query a specific node.
        
        if idx is empty, then the target is found and its item will be return
        else query its child by the first digit and pop the digit.

        Args:
            idx (string): node name

        Returns:
            list or node: if idx is empty, then the target is found and its item will be return. else query its child by the first digit and pop the digit.
            
        """
        if len(idx) > 1:
            return self(idx[0]).see(idx[1:])
        if idx == "*":
            return self.collect(return_bag=False)
        else:
            for i in self.items:
                if i[-1] == idx:
                    return [i]


class bagger():
    """
    A wrapper of sparser_node
    """

    def __init__(self):
        self.root = None
        self.fitted = False

    def fit(self, x, y=None):
        """
        Fit the string tree dictionary.
        Args:
            x (pandas.DataFrame): dataframe. should have .columns attribute
            y (pandas.Serise or 1D array, optional): label or target. This have no effects. Defaults to None.

        Returns:
            bagger: Fitted self.
        
        To do:
            transform
        """
        # sparse the column names
        self.root = sparser_node(x.columns, 0)

        # bagging
        self.template_x = x.copy()
        self.bags = self.root.bagging(self.template_x)

        self.fitted = True
        return self

    """
    def transform(self, x, y=None):
        
        if not self.fitted:
            raise "please call fit before calling transform."

        return self.template_x[self.bags], y
    """

    def fit_transform(self, x, y=None):
        """
        A functional stack of fit and calling the result

        Args:
            x (_type_): _description_
            y (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        self.fit(x, y)
        return self.template_x[self.bags], y  #self.transform(x, y)

    def see(self, idx):
        """
        A functional interface of sparser_node.see

        Args:
            idx (string): see sparser_node.see

        Returns:
            node or list: see sparser_node.see
        """
        return self.root.see(idx)

    def unbagging(self, name):
        """
        inverse transform of bagging

        Args:
            name (string): bagging name to query

        Returns:
            list: items in this bag.
        """
        return " ".join(self.see(name))
