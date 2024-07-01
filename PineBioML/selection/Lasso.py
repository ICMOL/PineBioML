import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import train_test_split
from . import SelectionPipeline, sample_weight


class Lasso_selection(SelectionPipeline):
    """
    Using Lasso (L1 penalty) regression as scoring method.  More specifically, L1 penalty will force feature weights to be zeros. 
    As the coefficient of penalty increases, more and more weights of features got killed and the important feature will remain.

    Lasso_selection will use grid search to find out when all weights vanish.

    Lasso_selection is scale sensitive in numerical and in result.

    """

    def __init__(self, unbalanced=True, objective="Regression"):
        """
        Args:
            unbalanced (bool, optional): False to imply class weight to samples. Defaults to True.
            objective (str, optional): one of {"Regression", "BinaryClassification"}
        """
        super().__init__()

        # parameters
        self.objective = objective
        if self.objective in ["regression", "Regression"]:
            self.regression = True
        else:
            self.regression = False
        self.da = 0.025  # d alpha
        self.blind = True
        self.upper_init = 50
        self.unbalanced = unbalanced
        self.name = "LassoLinear"

    def create_kernel(self, C):
        """
        Create diffirent kernel according to opjective.

        Args:
            C (float): The coefficient to L1 penalty.

        Returns:
            sklearn.linearmodel: a kernel of sklearn linearmodel
        """
        if self.regression:
            return Lasso(alpha=C)
        else:
            return LogisticRegression(penalty="l1",
                                      C=1 / C,
                                      solver="liblinear",
                                      random_state=142,
                                      class_weight="balanced")

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

        # train test split
        if not self.blind:
            X_train, X_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42)
        else:
            X_train = x
            y_train = y

        if self.unbalanced:
            weights = sample_weight(y_train)
        else:
            weights = np.ones_like(y_train)

        lassoes = []
        score = []
        # grid searching
        if self.regression:
            grids = np.arange(self.da, self.upper_init, self.da)
        else:
            grids = np.arange(self.upper_init, self.da, -self.da)

        for alpha in tqdm(grids):
            lassoes.append(self.create_kernel(C=alpha))
            lassoes[-1].fit(X_train, y_train, weights)
            alive = (lassoes[-1].coef_ != 0).sum()
            if not self.blind:
                score.append(((lassoes[-1].predict(X_test)
                               > 0.5) == y_test).mean())

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

    def __init__(self, unbalanced=True, objective="regression"):
        """
        Args:
            unbalanced (bool, optional): False to imply class weight to samples. Defaults to True.
            objective (str, optional): one of {"Regression", "BinaryClassification"}
        """
        super().__init__()
        self.upper_init = 1e+3
        self.lower_init = 1e-3
        self.objective = objective
        if self.objective in ["regression", "Regression"]:
            self.regression = True
        else:
            self.regression = False
        self.unbalanced = unbalanced
        self.blind = True
        self.name = "LassoLinear"

    def create_kernel(self, C):
        if self.regression:
            return Lasso(alpha=C)
        else:
            return LogisticRegression(penalty="l1",
                                      C=1 / C,
                                      solver="liblinear",
                                      random_state=142,
                                      class_weight="balanced")

    def Select(self, x, y, k):
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
        if not self.blind:
            X_train, X_test, y_train, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42)
        else:
            X_train = x
            y_train = y

        if self.unbalanced:
            weights = sample_weight(y_train)
        else:
            weights = np.ones_like(y_train)

        lassoes = []
        score = []
        # Bisection searching
        ### scaling x
        X_train = X_train / X_train.values.std()

        upper = self.upper_init
        lassoes.append(self.create_kernel(C=upper))
        lassoes[-1].fit(X_train, y_train, weights)
        if not self.blind:
            score.append(((lassoes[-1].predict(X_test)
                           > 0.5) == y_test).mean())
        upper_alive = (lassoes[-1].coef_ != 0).sum()
        #print(upper, upper_alive)

        lower = self.lower_init
        lassoes.append(self.create_kernel(C=lower))
        lassoes[-1].fit(X_train, y_train, weights)
        if not self.blind:
            score.append(((lassoes[-1].predict(X_test)
                           > 0.5) == y_test).mean())
        lower_alive = (lassoes[-1].coef_ != 0).sum()
        #print(lower, lower_alive)

        counter = 0
        while not lower_alive == k:
            alpha = (upper + lower) / 2
            lassoes.append(self.create_kernel(C=alpha))
            lassoes[-1].fit(X_train, y_train, weights)
            if not self.blind:
                score.append(((lassoes[-1].predict(X_test)
                               > 0.5) == y_test).mean())
            alive = (lassoes[-1].coef_ != 0).sum()
            #print(alive, alpha)

            if alive >= k:
                lower = alpha
                lower_alive = alive
            else:
                upper = alpha
                upper_alive = alive

            counter += 1
            if counter > 40:
                break

        coef = np.array([clr.coef_ for clr in lassoes])

        self.scores = pd.Series(np.abs(coef[-1]).flatten(),
                                index=x.columns,
                                name=self.name).sort_values(ascending=False)
        self.selected_score = self.scores.head(k)
        return self.selected_score


class multi_Lasso_selection(SelectionPipeline):
    """
    A stack of Lasso_bisection_selection. Because of collinearity, if there are a batch of featres with high corelation, only one of them will remain.
    That leads to diffirent behavior between select k features in a time and select k//n features in n times.
    """

    def __init__(self, objective="regression"):
        """
        Args:
            objective (str, optional): one of {"Regression", "BinaryClassification"}
        """
        super().__init__()
        self.name = "multi_Lasso"
        self.objective = objective

    def Select(self, x, y, k, n=5):
        """
        Select k//n features for n times, and then concatenate the results.

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.
            k (int): Number of feature to select. The result may less than k
            n (int, optional): Number of batch which splits k to select. Defaults to 10.

        Returns:
            pandas.Series: The score for k selected features. May less than k.
            
        """
        result = []
        if k == -1:
            k = x.shape[0]
        batch_size = k // n + 1

        for i in range(n):
            result.append(
                Lasso_bisection_selection(objective=self.objective).Select(
                    x, y, k=batch_size))
            x = x.drop(result[-1].index, axis=1)
        result = pd.concat(result)
        result = result - result.min()
        result.name = self.name

        self.selected_score = result.sort_values(ascending=False).head(k)
        return self.selected_score.copy()
