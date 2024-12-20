from . import Basic_tuner
from abc import abstractmethod
from typing import Literal

from joblib import parallel_backend

from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import numpy as np
from statsmodels.regression.linear_model import OLS

# Todo: Now loss(not metrics) only support mse, we need more like mae, mape, hinge... etc


class Regression_tuner(Basic_tuner):
    """
    A subclass of Basic_tuner for regression task.
    
    """

    def __init__(self,
                 n_try,
                 n_cv,
                 target,
                 validate_penalty,
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=None):
        super().__init__(n_try=n_try,
                         n_cv=n_cv,
                         target=target,
                         validate_penalty=validate_penalty,
                         kernel_seed=kernel_seed,
                         valid_seed=valid_seed,
                         optuna_seed=optuna_seed)
        """
        Args:
            n_try (int): The number of trials optuna should try.
            n_cv (int): The number of folds to execute cross validation evaluation in iteration of optuna optimization.
            target (str): The target of optuna optimization. Notice that is different from the training loss of model.
            validate_penalty (bool): True to penalty the overfitting by difference between training score and cv score.
            kernel_seed (int, optional): Random seed for model. Defaults to None.
            valid_seed (int, optional): Random seed for cross validation. Defaults to None.
            optuna_seed (int, optional): Random seed for optuna's hyperparameter sampling. Defaults to None.
        """

    def is_regression(self) -> bool:
        """
        Yes, it is for regression.

        Returns:
            bool: True
        """
        return True

    def reference(self) -> dict[str, str]:
        """
        This function will return reference of this method in python dict.    
        If you want to access it in PineBioML api document, then click on the    >Expand source code     

        Returns:
            dict[str, str]: a dict of reference.
        """
        return super().reference()

    @abstractmethod
    def name(self) -> str:
        """
        To be determined.

        Returns:
            str: Name of this tuner.
        """
        pass

    @abstractmethod
    def create_model(self, trial, default):
        """
        Create model based on default setting or optuna trial

        Args:
            trial (optuna.trial.Trial): optuna trial in this call.
            default (bool): To use default hyper parameter
            
        Returns :
            sklearn.base.BaseEstimator: A sklearn style model object.
        """
        pass


# linear model, elasticnet
class ElasticNet_tuner(Regression_tuner):
    """
    Tuning a elasic net regression.    
    [sklearn.linear_model.ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
    """

    def __init__(self,
                 n_try=25,
                 n_cv=5,
                 target="mse",
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=None,
                 validate_penalty=True):
        """
        Args:
            n_try (int, optional): The number of trials optuna should try. Defaults to 25.
            n_cv (int, optional): The number of folds to execute cross validation evaluation in iteration of optuna optimization. Defaults to 5.
            target (str, optional): The target of optuna optimization. Notice that is different from the training loss of model. Defaults to "mse".
            kernel_seed (int, optional): Random seed for model. Defaults to None.
            valid_seed (int, optional): Random seed for cross validation. Defaults to None.
            optuna_seed (int, optional): Random seed for optuna's hyperparameter sampling. Defaults to None.
            validate_penalty (bool, optional): True to penalty the overfitting by difference between training score and cv score. Defaults to True.
        """
        super().__init__(n_try=n_try,
                         n_cv=n_cv,
                         target=target,
                         kernel_seed=kernel_seed,
                         valid_seed=valid_seed,
                         optuna_seed=optuna_seed,
                         validate_penalty=validate_penalty)

    def name(self):
        return "ElasticNet"

    def reference(self) -> dict[str, str]:
        """
        This function will return reference of this method in python dict.    
        If you want to access it in PineBioML api document, then click on the    >Expand source code     

        Returns:
            dict[str, str]: a dict of reference.
        """
        refer = super().reference()
        refer[
            self.name() +
            " document"] = "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html"

        return refer

    def parms_range(self) -> dict:
        return {
            'alpha': ('alpha', "float", 1e-3, 1e+3),
            'l1_ratio': ('l1_ratio', "float", 0, 1)
        }

    def create_model(self, trial, default=False):
        parms = {}
        if not default:
            parms_to_tune = self.parms_range()
            for par in parms_to_tune:
                parms[par] = self.parms_range_sparser(trial,
                                                      parms_to_tune[par])
        enr = ElasticNet(**parms)
        return enr

    def summary(self):
        """
        It is the way I found to cram a sklearn regression result into the statsmodel regresion.    
        The only reason to do this is that statsmodel provides R-style summary.    
        """
        raise NotImplementedError
        sm_ols = OLS(self.y, self.x).fit_regularized(
            alpha=self.best_model.alpha,
            L1_wt=self.best_model.l1_ratio,
            start_params=self.best_model.coef_.flatten(),
            maxiter=0,
        )
        print(sm_ols.summary())


# RF
class RandomForest_tuner(Regression_tuner):
    """
    Tuning a random forest model.    
    [sklearn.ensemble.RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
    """

    def __init__(self,
                 using_oob=True,
                 n_try=50,
                 n_cv=5,
                 target="mse",
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=None,
                 validate_penalty=True):
        """

        Args:
            using_oob (bool, optional): Using out of bag score as validation. Defaults to True.
            n_try (int, optional): The number of trials optuna should try. Defaults to 50.
            n_cv (int, optional): The number of folds to execute cross validation evaluation in iteration of optuna optimization. Defaults to 5.
            target (str, optional): The target of optuna optimization. Notice that is different from the training loss of model. Defaults to "mse".
            kernel_seed (int, optional): Random seed for model. Defaults to None.
            valid_seed (int, optional): Random seed for cross validation. Defaults to None.
            optuna_seed (int, optional): Random seed for optuna's hyperparameter sampling. Defaults to None.
            validate_penalty (bool, optional): True to penalty the overfitting by difference between training score and cv score. Defaults to True.
        """
        super().__init__(n_try=n_try,
                         n_cv=n_cv,
                         target=target,
                         kernel_seed=kernel_seed,
                         valid_seed=valid_seed,
                         optuna_seed=optuna_seed,
                         validate_penalty=validate_penalty)

        self.using_oob = using_oob

    def name(self):
        return "RandomForest"

    def reference(self) -> dict[str, str]:
        """
        This function will return reference of this method in python dict.    
        If you want to access it in PineBioML api document, then click on the    >Expand source code     

        Returns:
            dict[str, str]: a dict of reference.
        """
        refer = super().reference()
        refer[
            self.name() +
            " document"] = "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html"
        refer[
            self.name() +
            " publication"] = "https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf"

        return refer

    def parms_range(self) -> dict:
        return {
            "n_estimators": ('n_estimators', "int", 32, 1024),
            'min_samples_leaf': ('min_samples_leaf', "int", 1, 32),
            'ccp_alpha': ('ccp_alpha', "float", 1e-4, 1e-1),
            'max_samples': ('max_samples', "float", 0.5, 0.9)
        }

    def create_model(self, trial, default=False):
        parms = {
            "bootstrap": self.using_oob,
            "oob_score": self.using_oob,
            "n_jobs": -1,
            "random_state": self.kernel_seed_tape[trial.number],
            "verbose": 0,
        }

        if not default:
            parms_to_tune = self.parms_range()
            for par in parms_to_tune:
                parms[par] = self.parms_range_sparser(trial,
                                                      parms_to_tune[par])

        rf = RandomForestRegressor(**parms)
        return rf

    def evaluate(self, trial, default=False):
        regressor_obj = self.create_model(trial, default)

        if self.using_oob:
            # oob predict
            with parallel_backend('loky'):
                regressor_obj.fit(self.x, self.y)
            y_pred = regressor_obj.oob_prediction_

            # oob score
            score = self.metric._score_func(self.y, y_pred)

            if not self.metric_great_better:
                # if not great is better, then multiply -1
                score *= -1
        else:
            # cv score
            cv = StratifiedKFold(
                n_splits=self.n_cv,
                shuffle=True,
                random_state=self.valid_seed_tape[trial.number])
            score = []

            for i, (train_ind, test_ind) in enumerate(cv.split(self.x,
                                                               self.y)):
                x_train = self.x.iloc[train_ind]
                y_train = self.y.iloc[train_ind]
                x_test = self.x.iloc[test_ind]
                y_test = self.y.iloc[test_ind]

                with parallel_backend('loky'):
                    regressor_obj.fit(self.x, self.y)

                test_score = self.metric(regressor_obj, x_test, y_test)
                train_score = self.metric(regressor_obj, x_train, y_train)

                if self.validate_penalty:
                    score.append(test_score + 0.1 * (test_score - train_score))
                else:
                    score.append(test_score)

            score = sum(score) / self.n_cv
        return score


# SVM
class SVM_tuner(Regression_tuner):
    """
    Tuning a support vector machine.    
    [sklearn.svm.SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
    """

    def __init__(self,
                 kernel: Literal["linear", "poly", "rbf", "sigmoid"] = "rbf",
                 n_try=25,
                 n_cv=5,
                 target="mse",
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=None,
                 validate_penalty=True):
        """
        Args:
            kernel (Literal[&quot;linear&quot;, &quot;poly&quot;, &quot;rbf&quot;, &quot;sigmoid&quot;], optional): This will be passed to the attribute of SVC: "kernel". Defaults to "rbf".
            n_try (int, optional): The number of trials optuna should try. Defaults to 25.
            n_cv (int, optional): The number of folds to execute cross validation evaluation in iteration of optuna optimization. Defaults to 5.
            target (str, optional): The target of optuna optimization. Notice that is different from the training loss of model. Defaults to "mse".
            kernel_seed (int, optional): Random seed for model. Defaults to None.
            valid_seed (int, optional): Random seed for cross validation. Defaults to None.
            optuna_seed (int, optional): Random seed for optuna's hyperparameter sampling. Defaults to None.
            validate_penalty (bool, optional): True to penalty the overfitting by difference between training score and cv score. Defaults to True.
        """
        super().__init__(n_try=n_try,
                         n_cv=n_cv,
                         target=target,
                         kernel_seed=kernel_seed,
                         valid_seed=valid_seed,
                         optuna_seed=optuna_seed,
                         validate_penalty=validate_penalty)
        self.kernel = kernel

    def name(self):
        return self.kernel + "-SVM"

    def reference(self) -> dict[str, str]:
        """
        This function will return reference of this method in python dict.    
        If you want to access it in PineBioML api document, then click on the    >Expand source code     

        Returns:
            dict[str, str]: a dict of reference.
        """
        refer = super().reference()
        refer[
            self.name() +
            " document"] = "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html"
        refer[
            self.name() +
            " sklearn backend"] = "http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf"
        refer[
            self.name() +
            " publication"] = "https://citeseerx.ist.psu.edu/doc_view/pid/42e5ed832d4310ce4378c44d05570439df28a393"

        return refer

    def parms_range(self) -> dict:
        # scaling penalty: https://scikit-learn.org/stable/auto_examples/svm/plot_svm_scale_c.html#sphx-glr-auto-examples-svm-plot-svm-scale-c-py
        return {
            'C': ('C', "float", 1e-4 * np.sqrt(self.n_sample),
                  1e+2 * np.sqrt(self.n_sample))
        }

    def create_model(self, trial, default=False):
        parms = {
            "kernel": self.kernel,
        }
        if not default:
            parms_to_tune = self.parms_range()
            for par in parms_to_tune:
                parms[par] = self.parms_range_sparser(trial,
                                                      parms_to_tune[par])
        svm = SVR(**parms)
        return svm


# XGboost
class XGBoost_tuner(Regression_tuner):
    """
    Tuning a XGBoost classifier model.    
    [xgboost.XGBClassifier](https://xgboost.readthedocs.io/en/stable/python/python_api.html)

    ToDo:    
        1. sample imbalance. (Done)
        2. early stop. (give up)
        3. efficiency (optuna.integration.XGBoostPruningCallback).

    """

    def __init__(self,
                 n_try=75,
                 n_cv=5,
                 target="mse",
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=None,
                 validate_penalty=True):
        """

        Args:
            n_try (int, optional): The number of trials optuna should try. Defaults to 75.
            n_cv (int, optional): The number of folds to execute cross validation evaluation in iteration of optuna optimization. Defaults to 5.
            target (str, optional): The target of optuna optimization. Notice that is different from the training loss of model. Defaults to "mse".
            kernel_seed (int, optional): Random seed for model. Defaults to None.
            valid_seed (int, optional): Random seed for cross validation. Defaults to None.
            optuna_seed (int, optional): Random seed for optuna's hyperparameter sampling. Defaults to None.
            validate_penalty (bool, optional): True to penalty the overfitting by difference between training score and cv score. Defaults to True.
        
        """
        super().__init__(n_try=n_try,
                         n_cv=n_cv,
                         target=target,
                         kernel_seed=kernel_seed,
                         valid_seed=valid_seed,
                         optuna_seed=optuna_seed,
                         validate_penalty=validate_penalty)

    def name(self):
        return "XGBoost"

    def reference(self) -> dict[str, str]:
        """
        This function will return reference of this method in python dict.    
        If you want to access it in PineBioML api document, then click on the    >Expand source code     

        Returns:
            dict[str, str]: a dict of reference.
        """
        refer = super().reference()
        refer[self.name() +
              " document"] = "https://xgboost.readthedocs.io/en/stable/"
        refer[
            self.name() +
            " publication"] = "https://dl.acm.org/doi/10.1145/2939672.2939785"

        return refer

    def parms_range(self) -> dict:
        return {
            "n_estimators": ('n_estimators', "int", 16, 256),
            "max_depth": ('max_depth', "int", 2, 16),
            "gamma": ('gamma', "float", 5e-2, 2e+1),
            "learning_rate": ('learning_rate', "float", 5e-2, 5e-1),
            "subsample": ('subsample', "float", 0.5, 1),
            "colsample_bytree": ('colsample_bytree', "float", 0.1, 0.9),
            "reg_lambda": ('reg_lambda', "float", 1e-2, 1e+1)
        }

    def create_model(self, trial, default=False):
        parms = {
            "n_jobs": None,
            "random_state": self.kernel_seed_tape[trial.number],
            "verbosity": 0
        }
        if not default:
            parms_to_tune = self.parms_range()
            for par in parms_to_tune:
                parms[par] = self.parms_range_sparser(trial,
                                                      parms_to_tune[par])

        xgb = XGBRegressor(**parms)
        return xgb


# lightGBM
class LighGBM_tuner(Regression_tuner):
    """
    Tuning a LighGBM classifier model.    
    [lightgbm.LGBMClassifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)     

    ToDo:    
        1. compare with optuna.integration.lightgbm.LightGBMTuner    
    """

    def __init__(self,
                 n_try=75,
                 n_cv=5,
                 target="mse",
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=None,
                 validate_penalty=True):
        """

        Args:
            n_try (int, optional): The number of trials optuna should try. Defaults to 75.
            n_cv (int, optional): The number of folds to execute cross validation evaluation in iteration of optuna optimization. Defaults to 5.
            target (str, optional): The target of optuna optimization. Notice that is different from the training loss of model. Defaults to "mse".
            kernel_seed (int, optional): Random seed for model. Defaults to None.
            valid_seed (int, optional): Random seed for cross validation. Defaults to None.
            optuna_seed (int, optional): Random seed for optuna's hyperparameter sampling. Defaults to None.
            validate_penalty (bool, optional): True to penalty the overfitting by difference between training score and cv score. Defaults to True.        
        
        """
        super().__init__(n_try=n_try,
                         n_cv=n_cv,
                         target=target,
                         kernel_seed=kernel_seed,
                         valid_seed=valid_seed,
                         optuna_seed=optuna_seed,
                         validate_penalty=validate_penalty)

    def name(self):
        return "LightGBM"

    def reference(self) -> dict[str, str]:
        """
        This function will return reference of this method in python dict.    
        If you want to access it in PineBioML api document, then click on the    >Expand source code     

        Returns:
            dict[str, str]: a dict of reference.
        """
        refer = super().reference()
        refer[
            self.name() +
            " document"] = "https://lightgbm.readthedocs.io/en/latest/index.html"
        refer[
            self.name() +
            " publication"] = "https://proceedings.neurips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf"

        return refer

    def parms_range(self) -> dict:
        return {
            "n_estimators": ('n_estimators', "int", 16, 256),
            "max_depth": ('max_depth', "int", 4, 16),
            "learning_rate": ('learning_rate', "float", 1e-2, 1),
            "subsample": ('subsample', "float", 0.5, 1),
            "colsample_bytree": ('colsample_bytree', "float", 0.1, 0.9),
            "reg_lambda": ('reg_lambda', "float", 5e-3, 1e+1)
        }

    def create_model(self, trial, default=False):
        parms = {
            "n_jobs": None,
            "random_state": self.kernel_seed_tape[trial.number],
            "verbosity": -1,
            "subsample_freq": 1
        }
        if not default:
            parms_to_tune = self.parms_range()
            for par in parms_to_tune:
                parms[par] = self.parms_range_sparser(trial,
                                                      parms_to_tune[par])

        lgbm = LGBMRegressor(**parms)
        return lgbm


# AdaBoost
class AdaBoost_tuner(Regression_tuner):
    """
    Tuning a AdaBoost regressor.    
    [sklearn.ensemble.AdaBoostRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)
    """

    def __init__(self,
                 n_try=25,
                 n_cv=5,
                 target="mse",
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=None,
                 validate_penalty=True):
        """
        Args:
            n_try (int, optional): The number of trials optuna should try. Defaults to 25.
            n_cv (int, optional): The number of folds to execute cross validation evaluation in iteration of optuna optimization. Defaults to 5.
            target (str, optional): The target of optuna optimization. Notice that is different from the training loss of model. Defaults to "mse".
            kernel_seed (int, optional): Random seed for model. Defaults to None.
            valid_seed (int, optional): Random seed for cross validation. Defaults to None.
            optuna_seed (int, optional): Random seed for optuna's hyperparameter sampling. Defaults to None.
            validate_penalty (bool, optional): True to penalty the overfitting by difference between training score and cv score. Defaults to True.
        """
        super().__init__(n_try=n_try,
                         n_cv=n_cv,
                         target=target,
                         kernel_seed=kernel_seed,
                         valid_seed=valid_seed,
                         optuna_seed=optuna_seed,
                         validate_penalty=validate_penalty)

    def name(self):
        return "AdaBoost"

    def reference(self) -> dict[str, str]:
        """
        This function will return reference of this method in python dict.    
        If you want to access it in PineBioML api document, then click on the    >Expand source code     

        Returns:
            dict[str, str]: a dict of reference.
        """
        refer = super().reference()
        refer[
            self.name() +
            " document"] = "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html"
        refer[
            self.name() +
            " publication: original version"] = "https://www.ee.columbia.edu/~sfchang/course/svia-F03/papers/freund95decisiontheoretic-adaboost.pdf"
        refer[
            self.name() +
            " publication: implemented version"] = "Drucker, 'Improving Regressors using Boosting Techniques', 1997"

        return refer

    def parms_range(self) -> dict:
        return {
            "n_estimators": ('n_estimators', "int", 8, 256),
            "learning_rate": ('learning_rate', "float", 1e-2, 1),
            "loss": ("loss", "category", ["linear", "square",
                                          "exponential"], None)
        }

    def create_model(self, trial, default=False):
        parms = {"random_state": self.kernel_seed_tape[trial.number]}
        if not default:
            parms_to_tune = self.parms_range()
            for par in parms_to_tune:
                parms[par] = self.parms_range_sparser(trial,
                                                      parms_to_tune[par])
        ada = AdaBoostRegressor(**parms)
        return ada


# DT
class DecisionTree_tuner(Regression_tuner):
    """
    Tuning a DecisionTree regressor.    
    [sklearn.tree.DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
    """

    def __init__(self,
                 n_try=25,
                 n_cv=5,
                 target="mse",
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=None,
                 validate_penalty=True):
        """
        Args:
            n_try (int, optional): The number of trials optuna should try. Defaults to 25.
            n_cv (int, optional): The number of folds to execute cross validation evaluation in iteration of optuna optimization. Defaults to 5.
            target (str, optional): The target of optuna optimization. Notice that is different from the training loss of model. Defaults to "mse".
            kernel_seed (int, optional): Random seed for model. Defaults to None.
            valid_seed (int, optional): Random seed for cross validation. Defaults to None.
            optuna_seed (int, optional): Random seed for optuna's hyperparameter sampling. Defaults to None.
            validate_penalty (bool, optional): True to penalty the overfitting by difference between training score and cv score. Defaults to True.
        """
        super().__init__(n_try=n_try,
                         n_cv=n_cv,
                         target=target,
                         kernel_seed=kernel_seed,
                         valid_seed=valid_seed,
                         optuna_seed=optuna_seed,
                         validate_penalty=validate_penalty)

    def name(self):
        return "DecisionTree"

    def reference(self) -> dict[str, str]:
        """
        This function will return reference of this method in python dict.    
        If you want to access it in PineBioML api document, then click on the    >Expand source code     

        Returns:
            dict[str, str]: a dict of reference.
        """
        refer = super().reference()
        refer[
            self.name() +
            " document"] = "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html"
        refer[
            self.name() +
            " publication"] = "https://www.taylorfrancis.com/books/mono/10.1201/9781315139470/classification-regression-trees-leo-breiman-jerome-friedman-olshen-charles-stone"

        return refer

    def parms_range(self) -> dict:
        return {
            "max_depth": ('max_depth', "int", 2, 16),
            "min_samples_split": ('min_samples_split', "int", 2, 32),
            "min_samples_leaf": ('min_samples_leaf', "int", 1, 16),
            "ccp_alpha": ('ccp_alpha', "float", 1e-3, 1e-1),
        }

    def create_model(self, trial, default=False):
        parms = {"random_state": self.kernel_seed_tape[trial.number]}
        if not default:
            parms_to_tune = self.parms_range()
            for par in parms_to_tune:
                parms[par] = self.parms_range_sparser(trial,
                                                      parms_to_tune[par])
        DT = DecisionTreeRegressor(**parms)
        return DT


# CatBoost
# KNN
