from . import SelectionPipeline
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss


class RF_selection(SelectionPipeline):
    """
    Using random forest to scoring (gini impurity / entropy) features.

    """

    def __init__(self, trees=1024 * 16, unbalanced=True, strategy="gini"):
        """
        Args:
            trees (int, optional): Number of trees. Defaults to 1024*16.
            strategy (str, optional): Scoring strategy, one of {"gini", "entropy"}. Defaults to "gini".
            unbalanced (bool, optional): True to imply class weight to samples. Defaults to True.
        """
        super().__init__()
        self.strategy = strategy
        if unbalanced:
            class_weight = "balanced"
        else:
            class_weight = None

        self.kernel = RandomForestClassifier(n_estimators=trees,
                                             n_jobs=-1,
                                             max_samples=0.75,
                                             class_weight=class_weight,
                                             criterion=strategy,
                                             verbose=1,
                                             ccp_alpha=1e-2,
                                             random_state=142)
        self.name = "RandomForest_" + self.strategy

    def Scoring(self, x, y=None):
        """
        Using random forest to scoring (gini impurity / entropy) features.

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.

        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.
        """
        self.kernel.fit(x, y)
        score = self.kernel.feature_importances_
        self.scores = pd.Series(score, index=x.columns,
                                name=self.name).sort_values(ascending=False)
        return self.scores.copy()


class oob_RFClassifier:
    """ 
    A random forest implement with out-of-bag evaluation. Boostrap subsampling strategy using Bernoulli sampling.
    """

    def __init__(self, trees=1024 * 8, unbalanced=True):
        """
        Args:
            trees (int, optional): Number of trees. Defaults to 1024*16.
            unbalanced (bool, optional): True to imply class weight to samples. Defaults to True.
        
        To do:
            Now is only for classification.
        """
        if unbalanced:
            class_weight = "balanced"
        else:
            class_weight = None
        self.subsampling_ratio = 0.7
        self.n_trees = trees
        self.tree_parms = {
            "criterion": "gini",
            "splitter": 'best',
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_weight_fraction_leaf": 0.0,
            "max_features": "sqrt",
            "class_weight": class_weight,
            "ccp_alpha": 1e-2
        }
        self.trees = {}

    def fit(self, x, y):
        """
        Training the random forest.

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods.
        """
        self.subsampling_table = pd.DataFrame(np.random.binomial(
            1, self.subsampling_ratio,
            size=(x.shape[0], self.n_trees)).astype(np.bool_),
                                              index=x.index)
        for i_th in tqdm(self.subsampling_table.columns):
            subsample = self.subsampling_table[i_th]
            sub_x = x.loc[subsample]
            sub_y = y.loc[subsample]

            self.trees[i_th] = DecisionTreeClassifier(**self.tree_parms).fit(
                sub_x, sub_y)

    def predict_prob(self, x):
        """
        Give a ratio that how many trees in the forest predict x as positive.

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.

        Returns:
            pandas.Series or a 1D array: ratio that how many trees in the forest predict x as positive. Defaults to None.
        """
        return pd.Series([t.predict(x) for t in self.trees.items],
                         index=x.index,
                         name="RF_prob").mean(axis=1)

    def oob_predict_prob(self, x):
        """
        x must be the training data.
        
        Give a ratio that how many trees from forest predict out-of-bag x as positive.

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.

        Returns:
            pandas.Series or a 1D array: ratio that how many trees from forest predict out-of-bag x as positive. Defaults to None.
        """
        if len(x) == len(self.subsampling_table):
            if not (x.index == self.subsampling_table.index).all():
                print(
                    "oob_predict_prob detect input x which diffirs from training"
                )
        else:
            print(
                "oob_predict_prob detect input x which diffirs from training")

        predicts = pd.Series([t.predict(x) for t in self.trees.items],
                             index=x.index,
                             name="RF_prob").divide
        oob_mask = np.logical_not(self.subsampling_table)
        predicts = predicts * oob_mask

        return predicts.sum(axis=1) / oob_mask.sum(axis=1)

    def evaluate(self, x, y, metric="ACC", oob=False):
        """
        Evaluate model using input x, y

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods.
            metric (str, optional): One of {acc, f1, bce, auc}. Defaults to "ACC".
            oob (bool, optional): True to use out of bag evaluate. Defaults to False.

        Returns:
            float: Evaluation result
        """
        if oob:
            y_pred = self.oob_predict_prob(x)
        else:
            y_pred = self.predict_prob(x)

        if metric in ["ACC", "acc", "accuracy"]:
            return accuracy_score(y, y_pred > 0.5)
        elif metric in ["f1", "F1", "f1_score", "F1_score"]:
            return f1_score(y, y_pred > 0.5)
        elif metric in ["BCE", "bce", "cross_entropy", "log_loss"]:
            return log_loss(y, y_pred)
        elif metric in ["AUC", "auc", "roc_auc", "ROC_AUC"]:
            return roc_auc_score(y, y_pred)
        else:
            print("Metric ", metric,
                  " not support! Please use one of acc, f1, bce or roc_auc.")
            return 0


class pcRF_selection(SelectionPipeline):
    """
    Expiriment method. PCA->RF->importance->inverse_PCA
    """

    def __init__(
        self,
        trees=512,
        unbalanced=True,
        strategy="permutation",
        factorize_method="PCA",
    ):
        """
        Args:
            trees (int, optional): Number of trees. Defaults to 512.
            unbalanced (bool, optional): _description_. Defaults to True.
            strategy (str, optional): Scoring strategy, one of {"gini", "entropy", "permutation"}. Defaults to "permutation".
            factorize_method (str, optional): One of {"PCA"}. Method to reduce dimension.  Defaults to "PCA".
                    
        """
        super().__init__()
        # remove colinearity
        #        if factorize_method == "NMF":
        #           self.fatorizer = NMF()
        #      else:
        #   self.fatorizer = PCA()

        if unbalanced:
            class_weight = "balanced"
        else:
            class_weight = None

        self.strategy = strategy
        #self.kernel = RandomForestClassifier(n_estimators = trees, bootstrap=True, oob_score=True, n_jobs=-1, class_weight = class_weight, verbose = 1)
        self.kernel = oob_RFClassifier(trees)
        self.name = "pcRandomForest_" + self.strategy

    def Scoring(self, x, y=None):
        """
        Using random forest to scoring (gini impurity / entropy) principal components.

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.

        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.
        """
        print("pc RF kernel fitting.")
        self.kernel.fit(x, y)

        n_repeat = 5
        print("Permutation evaluation start! n_repeat: ", n_repeat)
        score = pd.DataFrame(0,
                             index=x.columns,
                             columns=[i for i in range(n_repeat)])
        for i in range(n_repeat):  # repeat 5 times
            print("    repeat: ", i)
            for col in tqdm(x.columns):  # permutate each column
                x_permute = x.copy()
                x_permute.loc[:, col] = shuffle(x_permute[col]).values
                score.loc[col, i] = self.kernel.evaluate(x_permute,
                                                         y,
                                                         oob=True)

        score = score.mean(axis=1)

        return score

    def Select(self, x, y, k):
        """
        x->PCA->RF + oob +permutation importance -> inverse PCA -> feature importance

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.
            k (int): Number of feature to select. The result may less than k

        Returns:
            pandas.Series: The score for k selected features. May less than k.
        """

        # x should be a pd dataframe or a numpy array without missing value
        columns = x.columns
        index = x.index

        # remove colinearity
        # tune number of factor
        tmpca = PCA()
        pc_x = tmpca.fit_transform(x)
        accumucate_evr = 0
        threshold = 0.95
        i = 0
        for evr in tmpca.explained_variance_ratio_:
            accumucate_evr += evr
            i += 1
            if accumucate_evr > threshold:
                break
        self.n_pc = i

        # do the decomposition
        self.factorizer = PCA(self.n_pc)
        pc_x = pd.DataFrame(self.factorizer.fit_transform(x), index=index)

        # scoring
        pc_scores = self.scoring(pc_x, y)

        # revert importance of pricipal component to importance of feature via ratio of variance
        self.scores = pd.Series(
            pc_scores.dot(
                np.abs(self.factorizer.components_ *
                       self.factorizer.explained_variance_ratio_.reshape(
                           self.n_pc, 1))),  #
            index=columns,
            name=self.name).sort_values(ascending=False)

        selected_score = self.choose(self.scores, k)
        return selected_score
