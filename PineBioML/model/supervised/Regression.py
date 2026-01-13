from . import Basic_tuner
from abc import abstractmethod
from typing import Literal

from joblib import parallel_config

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping

import shap

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
                 TT_coef,
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=None):
        super().__init__(n_try=n_try,
                         n_cv=n_cv,
                         target=target,
                         validate_penalty=validate_penalty,
                         TT_coef=TT_coef,
                         kernel_seed=kernel_seed,
                         valid_seed=valid_seed,
                         optuna_seed=optuna_seed)

    def is_regression(self) -> bool:
        """
        Yes, it is for regression.

        Returns:
            bool: True
        """
        return True


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
                 validate_penalty=True,
                 TT_coef=0.1,
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=None):
        super().__init__(n_try=n_try,
                         n_cv=n_cv,
                         target=target,
                         validate_penalty=validate_penalty,
                         TT_coef=TT_coef,
                         kernel_seed=kernel_seed,
                         valid_seed=valid_seed,
                         optuna_seed=optuna_seed)

    def name(self):
        return "ElasticNet"

    def reference(self) -> dict[str, str]:
        refer = super().reference()
        refer[
            self.name() +
            " document"] = "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html"

        return refer

    def parms_range(self) -> dict:
        return {
            'alpha': ('alpha', "float", 1e-6, 1e+2),
            'l1_ratio': ('l1_ratio', "float", -0.5, 1.5)
        }

    def create_model(self, trial, default=False, training=False):
        parms = {"random_state": self.kernel_seed, "selection": "random"}
        if not default:
            parms_to_tune = self.parms_range()
            for par in parms_to_tune:
                if par == "l1_ratio":
                    parms[par] = np.clip(
                        self.parms_range_sparser(trial, parms_to_tune[par]), 0,
                        1)
                else:
                    parms[par] = self.parms_range_sparser(
                        trial, parms_to_tune[par])
            parms["random_state"] = self.kernel_seed_tape[trial.number]
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
    
    def _explainer(self, x):
        return shap.LinearExplainer(self.best_model, x)


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
                 validate_penalty=True,
                 TT_coef=0.1,
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=None):
        """
        Args:
            using_oob (bool, optional): Using out of bag score as validation. Defaults to True.
        """
        super().__init__(n_try=n_try,
                         n_cv=n_cv,
                         target=target,
                         validate_penalty=validate_penalty,
                         TT_coef=TT_coef,
                         kernel_seed=kernel_seed,
                         valid_seed=valid_seed,
                         optuna_seed=optuna_seed)

        self.using_oob = using_oob

    def name(self):
        return "RandomForest"

    def reference(self) -> dict[str, str]:
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
            "n_estimators": ('n_estimators', "int", 32, 512),
            'min_samples_leaf':
            ('min_samples_leaf', "int", 1, round(np.sqrt(self.n_sample) / 2)),
            'ccp_alpha':
            ('ccp_alpha', "float", 1e-2 / self.n_sample, 1e+2 / self.n_sample),
            'max_samples': ('max_samples', "float", 0.4, 0.8),
            "max_depth":
            ("max_depth", "int", round(np.log2(self.n_sample) / 2),
             int(np.log2(self.n_sample)) + 2)
        }

    def create_model(self, trial, default=False, training=False):
        parms = {
            "bootstrap": True,
            "oob_score": self.using_oob,
            "n_jobs": -1,
            "random_state": self.kernel_seed,
            "verbose": 0,
            "max_features": "sqrt",
        }

        if not default:
            parms_to_tune = self.parms_range()
            for par in parms_to_tune:
                parms[par] = self.parms_range_sparser(trial,
                                                      parms_to_tune[par])
            parms["random_state"] = self.kernel_seed_tape[trial.number]
        rf = RandomForestRegressor(**parms)
        return rf

    def evaluate(self, trial, default=False, training=False):
        regressor_obj = self.create_model(trial, default)

        if self.using_oob:
            # oob predict
            with parallel_config(backend='loky'):
                regressor_obj.fit(self.x, self.y)
            y_pred = regressor_obj.oob_prediction_

            # oob score
            score = self.metric._score_func(self.y, y_pred)

            if not self.metric_great_better:
                # if not great is better, then multiply -1
                score *= -1
        else:
            with parallel_config(backend='loky'):
                score = super().evaluate(trial=trial,
                                         default=default,
                                         training=training)
        return score
    
    def _explainer(self, x):
        return shap.TreeExplainer(self.best_model)


# SVM
class SVM_tuner(Regression_tuner):
    """
    Tuning a support vector machine.    
    [sklearn.svm.SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
    """

    def __init__(self,
                 n_try=25,
                 n_cv=5,
                 target="mse",
                 validate_penalty=True,
                 TT_coef=0.1,
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=None):
        super().__init__(n_try=n_try,
                         n_cv=n_cv,
                         target=target,
                         validate_penalty=validate_penalty,
                         TT_coef=TT_coef,
                         kernel_seed=kernel_seed,
                         valid_seed=valid_seed,
                         optuna_seed=optuna_seed)

    def name(self):
        return "SVM"

    def reference(self) -> dict[str, str]:
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
            "kernel":
            ('kernel', "category", ["linear", "poly", "rbf", "sigmoid"], None),
            'C': ('C', "float", 1e-3 * np.sqrt(self.n_sample),
                  1e+2 * np.sqrt(self.n_sample))
        }

    def create_model(self, trial, default=False, training=False):
        parms = {
            "kernel": "rbf",
        }
        if not default:
            parms_to_tune = self.parms_range()
            for par in parms_to_tune:
                parms[par] = self.parms_range_sparser(trial,
                                                      parms_to_tune[par])
        svm = SVR(**parms)
        return svm
    
    def _explainer(self, x):
        return shap.KernelExplainer(self.best_model.predict, x)


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
                 n_try=50,
                 n_cv=5,
                 target="mse",
                 validate_penalty=True,
                 TT_coef=0.1,
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=None):
        super().__init__(n_try=n_try,
                         n_cv=n_cv,
                         target=target,
                         validate_penalty=validate_penalty,
                         TT_coef=TT_coef,
                         kernel_seed=kernel_seed,
                         valid_seed=valid_seed,
                         optuna_seed=optuna_seed)

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
            "n_estimators": ('n_estimators', "int", 4, 256),
            "max_depth":
            ("max_depth", "int", round(np.log2(self.n_sample) / 2),
             int(np.log2(self.n_sample)) + 2),
            "gamma": ('gamma', "float", 1e-4, 1e-2),
            "min_child_weight": ("min_child_weight", "float", 1,
                                 round(np.sqrt(self.n_sample) / 2)),
            "learning_rate": ('learning_rate', "float", 1e-1, 1.),
            "subsample": ('subsample', "float", 0.5, 1.),
            "colsample_bytree": ('colsample_bytree', "float", 0.5, 1),
            "reg_lambda": ('reg_lambda', "float", 1e-3, 1),
            "reg_alpha": ('reg_alpha', "float", 1e-4, 1.)
        }

    def create_model(self, trial, default=False, training=False):
        parms = {
            "n_jobs": None,
            "random_state": self.kernel_seed,
            "verbosity": 0
        }
        if not default:
            parms_to_tune = self.parms_range()
            for par in parms_to_tune:
                parms[par] = self.parms_range_sparser(trial,
                                                      parms_to_tune[par])
            parms["random_state"] = self.kernel_seed_tape[trial.number]

            if training:
                # Adding early stopping callbacks
                parms["early_stopping_rounds"] = round(
                    parms["n_estimators"] * 0.1) + 2
            else:
                # not training, then overide the n_estimators by trial's early stopping point.
                parms["n_estimators"] = self.stop_points[trial.number]

        xgb = XGBRegressor(**parms)
        return xgb

    def using_earlystopping(self):
        return True

    def clr_best_iteration(self, classifier):
        return classifier.best_iteration

    def optimize_fit(self, clr, train_data, sample_weight, valid_data):
        train_x, train_y = train_data

        return clr.fit(train_x,
                       train_y,
                       sample_weight=sample_weight,
                       eval_set=[valid_data],
                       verbose=False)
    
    def _explainer(self, x):
        return shap.TreeExplainer(self.best_model)


# lightGBM
class LightGBM_tuner(Regression_tuner):
    """
    Tuning a LighGBM classifier model.    
    [lightgbm.LGBMClassifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)     

    ToDo:    
        1. compare with optuna.integration.lightgbm.LightGBMTuner    
    """

    def __init__(self,
                 n_try=50,
                 n_cv=5,
                 target="mse",
                 validate_penalty=True,
                 TT_coef=0.1,
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=None):
        super().__init__(n_try=n_try,
                         n_cv=n_cv,
                         target=target,
                         validate_penalty=validate_penalty,
                         TT_coef=TT_coef,
                         kernel_seed=kernel_seed,
                         valid_seed=valid_seed,
                         optuna_seed=optuna_seed)

    def name(self):
        return "LightGBM"

    def reference(self) -> dict[str, str]:
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
            "n_estimators": ('n_estimators', "int", 4, 256),
            "max_depth":
            ("max_depth", "int", round(np.log2(self.n_sample) / 2),
             int(np.log2(self.n_sample)) + 2),
            "min_child_samples":
            ("min_child_samples", "int", 1, int(np.sqrt(self.n_sample) / 2)),
            "learning_rate": ('learning_rate', "float", 1e-2, 1.),
            "subsample": ('subsample', "float", 0.5, 1.),
            "colsample_bytree": ('colsample_bytree', "float", 0.5, 1.),
            "reg_lambda": ('reg_lambda', "float", 1e-3, 1),
            "reg_alpha": ('reg_alpha', "float", 1e-4, 1.)
        }

    def create_model(self, trial, default=False, training=False):
        parms = {
            "n_jobs": None,
            "random_state": self.kernel_seed,
            "verbosity": -1,
            "subsample_freq": 1
        }
        if not default:
            parms_to_tune = self.parms_range()
            for par in parms_to_tune:
                parms[par] = self.parms_range_sparser(trial,
                                                      parms_to_tune[par])
            parms["random_state"] = self.kernel_seed_tape[trial.number]
            if not training:
                # not training, then overide the n_estimators by trial's early stopping point.
                parms["n_estimators"] = self.stop_points[trial.number]

        lgbm = LGBMRegressor(**parms)
        return lgbm

    def using_earlystopping(self):
        return True

    def clr_best_iteration(self, classifier):
        return classifier.best_iteration_

    def optimize_fit(self, clr, train_data, sample_weight, valid_data):
        train_x, train_y = train_data

        return clr.fit(train_x,
                       train_y,
                       sample_weight=sample_weight,
                       eval_set=[valid_data],
                       callbacks=[
                           early_stopping(round(clr.n_estimators * 0.1) + 2,
                                          verbose=False)
                       ])
    
    def _explainer(self, x):
        return shap.TreeExplainer(self.best_model)


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
                 validate_penalty=True,
                 TT_coef=0.1,
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=None):
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
                         validate_penalty=validate_penalty,
                         TT_coef=TT_coef,
                         kernel_seed=kernel_seed,
                         valid_seed=valid_seed,
                         optuna_seed=optuna_seed)

    def name(self):
        return "AdaBoost"

    def reference(self) -> dict[str, str]:
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
            "n_estimators": ('n_estimators', "int", 4, 64),
            "learning_rate": ('learning_rate', "float", 1e-2, 1.),
            "loss": ("loss", "category", ["linear", "square",
                                          "exponential"], None)
        }

    def create_model(self, trial, default=False, training=False):
        parms = {"random_state": self.kernel_seed}
        if not default:
            parms_to_tune = self.parms_range()
            for par in parms_to_tune:
                parms[par] = self.parms_range_sparser(trial,
                                                      parms_to_tune[par])
            parms["random_state"] = self.kernel_seed_tape[trial.number]
        ada = AdaBoostRegressor(**parms)
        return ada

    def _explainer(self, x):
        return shap.KernelExplainer(self.best_model.predict, x)

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
                 validate_penalty=True,
                 TT_coef=0.1,
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=None):
        super().__init__(n_try=n_try,
                         n_cv=n_cv,
                         target=target,
                         validate_penalty=validate_penalty,
                         TT_coef=TT_coef,
                         kernel_seed=kernel_seed,
                         valid_seed=valid_seed,
                         optuna_seed=optuna_seed)

    def name(self):
        return "DecisionTree"

    def reference(self) -> dict[str, str]:
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
            "max_depth":
            ("max_depth", "int", round(np.log2(self.n_sample) / 2),
             int(np.log2(self.n_sample)) + 2),
            "min_samples_leaf":
            ('min_samples_leaf', "int", 1, round(np.sqrt(self.n_sample) / 2)),
            'ccp_alpha':
            ('ccp_alpha', "float", 1e-2 / self.n_sample, 1e+2 / self.n_sample),
        }

    def create_model(self, trial, default=False, training=False):
        parms = {"random_state": self.kernel_seed}
        if not default:
            parms_to_tune = self.parms_range()
            for par in parms_to_tune:
                parms[par] = self.parms_range_sparser(trial,
                                                      parms_to_tune[par])
            parms["random_state"] = self.kernel_seed_tape[trial.number]
        DT = DecisionTreeRegressor(**parms)
        return DT
    
    def _explainer(self, x):
        return shap.TreeExplainer(self.best_model)


# CatBoost
# KNN
