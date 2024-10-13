from . import SelectionPipeline

import pandas as pd
import numpy as np
from tqdm import tqdm

from joblib import parallel_backend

from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.svm import LinearSVR
import xgboost as xgb
import lightgbm as lgbm


class Lasso_selection(SelectionPipeline):
    """
    Using Lasso (L1 penalty) regression as scoring method.  More specifically, L1 penalty will force feature weights to be zeros. 
    As the coefficient of penalty increases, more and more weights of features got killed and the important feature will remain.

    Lasso_selection will use grid search to find out when all weights vanish.

    Lasso_selection is scale sensitive in numerical and in result.

    """

    def __init__(self, k):
        """
        Args:
            unbalanced (bool, optional): False to imply class weight to samples. Defaults to True.
            objective (str, optional): one of {"Regression", "BinaryClassification"}
        """
        super().__init__(k=k)

        # parameters
        self.da = 0.25  # d alpha
        self.upper_init = 50
        self.name = "LassoLinear"

    def create_kernel(self, C):
        """
        Create diffirent kernel according to opjective.

        Args:
            C (float): The coefficient to L1 penalty.

        Returns:
            sklearn.linearmodel: a kernel of sklearn linearmodel
        """
        return Lasso(alpha=C)

    def Scoring(self, x, y=None):
        """
        Using Lasso (L1 penalty) regression as scoring method.  More specifically, L1 penalty will force feature weights to be zeros. 
        As the coefficient of penalty increases, more and more weights of features got killed and the important feature will remain.

        Lasso_selection will use grid search to find out when all weights vanish.

         Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.

        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.
        
        To do:
            kfold validation performance threshold.
        
        """

        X_train = x.copy()
        y_train = y.copy()

        if self.k == -1:
            self.k = x.shape[0]

        lassoes = []
        # grid searching
        grids = np.arange(self.da, self.upper_init, self.da)

        for alpha in tqdm(grids):
            lassoes.append(self.create_kernel(C=alpha))
            lassoes[-1].fit(X_train, y_train)
            alive = (lassoes[-1].coef_ != 0).sum()

            if alive < 1:
                print("all coefficient are dead, terminated.")
                break

        coef = np.array([clr.coef_ for clr in lassoes]).flatten()

        self.scores = pd.Series(np.logical_not(coef == 0).sum(axis=0) *
                                self.da,
                                index=x.columns,
                                name=self.name).sort_values(ascending=False)
        return self.scores.copy()


class Lasso_bisection_selection(SelectionPipeline):
    """
    Using Lasso (L1 penalty) regression as scoring method.  More specifically, L1 penalty will force feature weights to be zeros. 
    As the coefficient of penalty increases, more and more weights of features got killed and the important feature will remain.

    Lasso_bisection_selection will use binary search to find out when all weights vanish.
    
    The trace of weight vanishment is not support.

    """

    def __init__(self, k):
        """
        Args:
            unbalanced (bool, optional): False to imply class weight to samples. Defaults to True.
            objective (str, optional): one of {"Regression", "BinaryClassification"}
        """
        super().__init__(k=k)
        self.upper_init = 1e+5
        self.lower_init = 1e-5
        self.name = "LassoLinear"

    def create_kernel(self, C):
        return Lasso(alpha=C)

    def Select(self, x, y):
        """
        Using Lasso (L1 penalty) regression as scoring method.  More specifically, L1 penalty will force feature weights to be zeros. 
        As the coefficient of penalty increases, more and more weights of features got killed and the important feature will remain.

        Lasso_bisection_selection will use binary search to find out when all weights vanish.

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.
            k (int): Number of feature to select. The result may less than k

        Returns:
            pandas.Series: The score for k selected features. May less than k.
        """

        # train test split
        X_train = x.copy()
        y_train = y.copy()

        if self.k == -1:
            self.k = x.shape[0]

        lassoes = []
        # Bisection searching
        ### standardize x
        X_train = X_train / X_train.values.std()

        upper = self.upper_init
        lassoes.append(self.create_kernel(C=upper))
        lassoes[-1].fit(X_train, y_train)
        upper_alive = (self.coef_to_importance(lassoes[-1].coef_) != 0).sum()
        #print(upper, upper_alive)

        lower = self.lower_init
        lassoes.append(self.create_kernel(C=lower))
        lassoes[-1].fit(X_train, y_train)
        lower_alive = (self.coef_to_importance(lassoes[-1].coef_) != 0).sum()
        #print(lower, lower_alive)

        counter = 0
        while not lower_alive == self.k:
            alpha = (upper + lower) / 2
            lassoes.append(self.create_kernel(C=alpha))
            lassoes[-1].fit(X_train, y_train)
            alive = (self.coef_to_importance(lassoes[-1].coef_) != 0).sum()
            #print(alive, alpha)

            if alive >= self.k:
                lower = alpha
                lower_alive = alive
            else:
                upper = alpha
                upper_alive = alive

            counter += 1
            if counter > 40:
                break

        coef = np.array([clr.coef_ for clr in lassoes])

        self.scores = pd.Series(self.coef_to_importance(coef[-1]),
                                index=x.columns,
                                name=self.name).sort_values(ascending=False)
        self.selected_score = self.scores.head(self.k)
        return self.selected_score

    def coef_to_importance(self, coef):
        return np.linalg.norm(coef, ord=2, axis=0)


class multi_Lasso_selection(SelectionPipeline):
    """
    A stack of Lasso_bisection_selection. Because of collinearity, if there are a batch of featres with high corelation, only one of them will remain.
    That leads to diffirent behavior between select k features in a time and select k//n features in n times.
    """

    def __init__(self, k):
        """
        Args:
            objective (str, optional): one of {"Regression", "BinaryClassification"}
        """
        super().__init__(k=k)
        self.name = "multi_Lasso"

    def Select(self, x, y, n=5):
        """
        Select k//n features for n times, and then concatenate the results.

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.
            n (int, optional): Number of batch which splits k to select. Defaults to 10.

        Returns:
            pandas.Series: The score for k selected features. May less than k.
            
        """
        result = []
        if self.k == -1:
            self.k = x.shape[0]
        batch_size = self.k // n + 1

        for i in range(n):
            result.append(Lasso_bisection_selection(k=batch_size).Select(x, y))
            x = x.drop(result[-1].index, axis=1)
            if x.shape[1] == 0:
                break
        result = pd.concat(result)
        result = result - result.min()
        result.name = self.name

        self.selected_score = result.sort_values(ascending=False).head(self.k)
        return self.selected_score.copy()


class SVM_selection(SelectionPipeline):
    """
    Using the support vector of linear support vector classifier as scoring method.

    SVM_selection is scale sensitive in result.

    <<Feature Ranking Using Linear SVM>> section 3.2

    """

    def __init__(self, k):
        super().__init__(k=k)
        self.kernel = LinearSVR(random_state=142)
        self.name = "SVM"

    def Scoring(self, x, y=None):
        """
        Using the support vector of linear support vector classifier as scoring method.

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.

        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.
        """
        self.kernel.fit(x, y)
        svm_weights = np.abs(self.kernel.coef_).sum(axis=0)
        svm_weights /= svm_weights.sum()

        self.scores = pd.Series(svm_weights, index=x.columns,
                                name=self.name).sort_values(ascending=False)
        return self.scores.copy()


# ToDo: CART
'''
class DT_selection(SelectionPipeline):
    """
    A child class of SelectionPipeline .

    Using Decision stump (a single Decision tree) to scoring features.
    """

    def __init__(self, k, bins=10, q=0.05, strategy="c45"):
        """
        Args:
            bins (int, optional): Bins to esimate data distribution entropy. Defaults to 10.
            q (float, optional): Clip data values out of [q, 1-q] percentile to reduce the affect of outliers while estimate entropy. Defaults to 0.05.
            strategy (str, optional): One of {"id3", "c45"}. The strategy to build decision tree. Defaults to "c45".
        """
        super().__init__(k=k)
        self.bins = bins - 1
        self.q = q
        self.strategy = strategy
        self.name = "DT_score_" + self.strategy

    def Scoring(self, x, y=None):
        """
        Using Decision stump (a single Decision tree) to scoring features. Though, single layer stump is equivalent to compare the id3/c4.5 score directly.

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.

        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.
        """
        upper = x.quantile(1 - self.q)
        lower = x.quantile(self.q)
        #print(upper, lower)
        normed = (x - lower) / (upper - lower)
        normed = normed.clip(0, 1)
        bin_idx = (normed * self.bins - 0.5).round().astype(np.int32)
        columns = bin_idx.columns

        bin_idx["label"] = y

        scores = []
        for i in tqdm(columns):
            feature_hists = bin_idx[[i, "label"]].groupby(i)
            feature_entropy = feature_hists.apply(self.entropy)
            feature_size = feature_hists.apply(len) + 1e-3

            info = (feature_entropy / feature_size).sum()
            gain = 0 - info
            if self.strategy == "c45":
                freq = bin_idx[i].value_counts()
                p = freq / freq.sum()
                split_info = -p * np.log(p)
                gain /= split_info.sum()
            scores.append(gain)
        scores = pd.Series(scores, index=columns,
                           name=self.name).sort_values(ascending=False)
        scores = scores - scores.min()
        return scores

    def entropy(self, x):
        """
        Estimate entropy

        Args:
            x (pandas.DataFrame): data with bined label

        Returns:
            float: entropy
        """
        label_nums = x["label"].value_counts()
        label_prop = label_nums / label_nums.sum()

        entropy = -(label_prop * np.log(label_prop + 1e-6)).sum()

        return entropy
'''


class RF_selection(SelectionPipeline):
    """
    Using random forest to scoring (gini impurity / entropy) features.

    """

    def __init__(self, k, trees=1024, strategy="squared_error"):
        """
        Args:
            trees (int, optional): Number of trees. Defaults to 1024*16.
            strategy (str, optional): Scoring strategy, one of {“squared_error”, “absolute_error”, “friedman_mse”, “poisson” }.
            unbalanced (bool, optional): True to imply class weight to samples. Defaults to True.
        """
        super().__init__(k=k)
        self.strategy = strategy

        self.kernel = RandomForestRegressor(n_estimators=trees,
                                            n_jobs=-1,
                                            max_samples=0.75,
                                            criterion=strategy,
                                            verbose=0,
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
        with parallel_backend('loky'):
            self.kernel.fit(x, y)
        score = self.kernel.feature_importances_
        self.scores = pd.Series(score, index=x.columns,
                                name=self.name).sort_values(ascending=False)
        return self.scores.copy()


class XGboost_selection(SelectionPipeline):
    """
    Using XGboost to scoring (gini impurity / entropy) features.

    Warning: If data is too easy, boosting methods is difficult to give score to all features.
    """

    def __init__(self, k):
        """
        Args:
            unbalanced (bool, optional): True to imply class weight to samples. Defaults to False.
        """
        super().__init__(k=k)

        self.kernel = xgb.XGBRegressor(random_state=142, subsample=0.7)
        self.name = "XGboost"

    def Scoring(self, x, y=None):
        """
        Using XGboost to scoring (gini impurity / entropy) features.

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


class Lightgbm_selection(SelectionPipeline):
    """
    Using Lightgbm to scoring (gini impurity / entropy) features. 

    Warning: If data is too easy, boosting methods is difficult to give score to all features.
    """

    def __init__(self, k, unbalanced=True):
        """
        Args:
            unbalanced (bool, optional): True to imply class weight to samples. Defaults to False.
        """
        super().__init__(k=k)
        self.unbalanced = unbalanced

        self.kernel = lgbm.LGBMRegressor(learning_rate=0.01,
                                         random_state=142,
                                         subsample=0.7,
                                         subsample_freq=1,
                                         verbosity=-1)
        self.name = "Lightgbm"

    def Scoring(self, x, y=None):
        """
        Using Lightgbm to scoring (gini impurity / entropy) features.

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


class AdaBoost_selection(SelectionPipeline):
    """
    Using AdaBoost to scoring (gini impurity / entropy) features.

    Warning: If data is too easy, boosting methods is difficult to give score to all features.
    """

    def __init__(self, k, unbalanced=True, n_iter=128, learning_rate=0.01):
        """
        Args:
            n_iter (int, optional): Number of trees also number of iteration to boost. Defaults to 64.
            learning_rate (float, optional): boosting learning rate. Defaults to 0.01.
            unbalanced (bool, optional): True to imply class weight to samples. Defaults to False.
        """
        super().__init__(k=k)
        self.unbalanced = unbalanced
        self.kernel = AdaBoostRegressor(
            n_estimators=n_iter,
            learning_rate=learning_rate,
            random_state=142,
        )
        self.name = "AdaBoost" + str(n_iter)

    def Scoring(self, x, y=None):
        """
        Using AdaBoost to scoring (gini impurity / entropy) features.

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


class essemble_selector(SelectionPipeline):
    """
    A functional stack of diffirent methods.
    
    """

    def __init__(self, k=-1, RF_trees=1024, z_importance_threshold=None):
        """

        Args:

        """
        self.z_importance_threshold = z_importance_threshold
        self.k = k

        self.kernels = {
            "RF_gini": RF_selection(k=k, trees=RF_trees),
            "Lasso_Bisection": Lasso_bisection_selection(k=k),
            "multi_Lasso": multi_Lasso_selection(k=k),
            "SVM": SVM_selection(k=k),
            "AdaBoost": AdaBoost_selection(k=k),
            "XGboost": XGboost_selection(k=k),
            "Lightgbm": Lightgbm_selection(k=k)
        }

    def Select(self, x, y):
        """
        Calling all the methods in kernel sequancially.

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.
            k (int): Number of feature to select. The result may less than k

        Returns:
            pandas.Series: The concatenated results. Top k (may less than k) important feature from diffient methods.
        """
        results = []
        for method in self.kernels:
            print("Using ", method, " to select.")
            results.append(self.kernels[method].Select(x.copy(), y))
            print(method, " is done.\n")

        name = pd.concat([pd.Series(i.index, name=i.name) for i in results],
                         axis=1)
        importance = pd.concat(results, axis=1)
        self.selected_score = importance
        return name, importance

    def fit(self, x, y):
        """
        sklearn api

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods.
        """
        name, importance = self.Select(x, y)

        return self

    def transform(self, x):
        z_scores = (self.selected_score - self.selected_score.mean()) / (
            self.selected_score.std() + 1e-4)
        z_scores = z_scores.mean(axis=1).sort_values(ascending=False)

        if self.z_importance_threshold is None:
            return x[z_scores.index[:self.k]]
        else:
            return x[z_scores[z_scores > self.z_importance_threshold].index]

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)

    def plotting(self):
        for method in self.kernels:
            self.kernels[method].plotting()
