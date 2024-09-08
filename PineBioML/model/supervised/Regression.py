from . import Basic_tuner

from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import numpy as np
from statsmodels.regression.linear_model import OLS

# ToDo: sample weight
# ToDo: now we only support mse now, we have to add more loss.
# ToDo: random seed for optuna, cross validation and model. Currently, cross validation and model shared the same seed
# ToDo: optuna pruner
#       see section Acticating Pruners in https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html


# linear model, elasticnet
class ElasticNet_tuner(Basic_tuner):
    """
    Tuning a elasic net regression.    
    [sklearn.linear_model.ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
    """

    def __init__(self,
                 n_try=25,
                 target="neg_mean_squared_error",
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=71,
                 validate_penalty=True):
        """
        Args:
            n_try (int, optional): Times to try. Defaults to 25.    
            target (str, optional): The target of hyperparameter tuning. It will pass to sklearn.metrics.get_scorer . Using sklearn.metrics.get_scorer_names() to list available metrics. Defaults to "neg_mean_squared_error" (mse score).    
            kernel_seed (int): random seed for model kernel. 
            valid_seed (int): random seed for cross validation
            optuna_seed (int, optional): random seed for optuna. Defaults to 71.     
        """
        super().__init__(n_try=n_try,
                         target=target,
                         kernel_seed=kernel_seed,
                         valid_seed=valid_seed,
                         optuna_seed=optuna_seed,
                         validate_penalty=validate_penalty)

    def create_model(self, trial, default=False):
        if default:
            parms = {"verbose": 0}
        else:
            parms = {
                "alpha": trial.suggest_float('alpha', 1e-3, 1e+3, log=True),
                "l1_ratio": trial.suggest_float('l1_ratio', 0, 1),
                "verbose": 0
            }
        enr = ElasticNet(**parms)
        return enr

    def summary(self):
        """
        It is the way I found to cram a sklearn regression result into the statsmodel regresion.    
        The only reason to do this is that statsmodel provides R-style summary.    
        """
        sm_ols = OLS(self.y, self.x).fit_regularized(
            disp=False,
            alpha=self.best_model.alpha,
            L1_wt=self.best_model.l1_ratio,
            start_params=self.best_model.coef_.flatten(),
            maxiter=0,
            warn_convergence=False)
        print(sm_ols.summary())


# RF
class RandomForest_tuner(Basic_tuner):
    """
    Tuning a random forest model.    
    [sklearn.ensemble.RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)
    """

    def __init__(self,
                 using_oob=True,
                 n_try=50,
                 target="neg_mean_squared_error",
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=71,
                 validate_penalty=True):
        """

        Args:
            using_oob (bool, optional): Using out of bag score as validation. Defaults to True.
            n_try (int, optional): Times to try. Defaults to 50.
            target (str, optional): The target of hyperparameter tuning. It will pass to sklearn.metrics.get_scorer . Using sklearn.metrics.get_scorer_names() to list available metrics. Defaults to "neg_mean_squared_error" (mse score).    
            kernel_seed (int): random seed for model kernel. 
            valid_seed (int): random seed for cross validation
            optuna_seed (int, optional): random seed for optuna. Defaults to 71.     
        """
        super().__init__(n_try=n_try,
                         target=target,
                         kernel_seed=kernel_seed,
                         valid_seed=valid_seed,
                         optuna_seed=optuna_seed,
                         validate_penalty=validate_penalty)

        self.using_oob = using_oob

    def create_model(self, trial, default=False):
        if default:
            parms = {
                "bootstrap": self.using_oob,
                "oob_score": self.using_oob,
                "n_jobs": 1,  # faster than -1.  omg
                "random_state": self.kernel_seed,
                "verbose": 0,
            }
        else:
            parms = {
                "n_estimators":
                trial.suggest_int('n_estimators', 32, 1024, log=True),
                "max_depth":
                trial.suggest_int('max_depth', 2, 16, log=True),
                #"min_samples_split":
                #trial.suggest_int('min_samples_split', 2, 16, log=True),
                "min_samples_leaf":
                trial.suggest_int('min_samples_leaf', 1, 16, log=True),
                "ccp_alpha":
                trial.suggest_float('ccp_alpha', 1e-3, 1e-1, log=True),
                "max_samples":
                trial.suggest_float('max_samples', 0.5, 0.9, log=True),
                "bootstrap":
                self.using_oob,
                "oob_score":
                self.using_oob,
                "n_jobs":
                1,  # faster than -1.  omg
                "random_state":
                self.kernel_seed_tape[trial.number],
                "verbose":
                0,
            }

        rf = RandomForestRegressor(**parms)
        return rf

    def evaluate(self, trial, default=False):
        classifier_obj = self.create_model(trial, default)

        if self.using_oob:
            # oob predict
            classifier_obj.fit(self.x, self.y)
            y_pred = classifier_obj.oob_decision_function_[:, 1]

            # oob score
            score = self.metric._score_func(self.y, y_pred)
        else:
            # cv score
            score = cross_val_score(
                classifier_obj,
                self.x,
                self.y,
                cv=StratifiedKFold(
                    n_splits=5,
                    shuffle=True,
                    random_state=self.valid_seed_tape[trial.number]),
                scoring=self.metric,
                n_jobs=-1)
            score = score.mean()
        return score


# SVM
class SVM_tuner(Basic_tuner):
    """
    Tuning a support vector machine.    
    [sklearn.svm.SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
    """

    def __init__(self,
                 kernel="rbf",
                 n_try=25,
                 target="neg_mean_squared_error",
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=71,
                 validate_penalty=True):
        """
        Args:
            kernel (str, optional): This will be passed to the attribute of SVR: "kernel". Defaults to "rbf".
            n_try (int, optional): Times to try. Defaults to 50.
            target (str, optional): The target of hyperparameter tuning. It will pass to sklearn.metrics.get_scorer . Using sklearn.metrics.get_scorer_names() to list available metrics. Defaults to "neg_mean_squared_error" (mse score).    
            kernel_seed (int): random seed for model kernel. 
            valid_seed (int): random seed for cross validation
            optuna_seed (int, optional): random seed for optuna. Defaults to 71.     
        """
        super().__init__(n_try=n_try,
                         target=target,
                         kernel_seed=kernel_seed,
                         valid_seed=valid_seed,
                         optuna_seed=optuna_seed,
                         validate_penalty=validate_penalty)
        self.kernel = kernel  # rbf, linear, poly, sigmoid

    def create_model(self, trial, default=False):
        if default:
            parms = {
                "kernel": self.kernel,
                "random_state": self.kernel_seed,
            }
        else:
            # scaling penalty: https://scikit-learn.org/stable/auto_examples/svm/plot_svm_scale_c.html#sphx-glr-auto-examples-svm-plot-svm-scale-c-py
            parms = {
                "C":
                trial.suggest_float('C',
                                    1e-4 * np.sqrt(self.n_sample),
                                    1e+2 * np.sqrt(self.n_sample),
                                    log=True),
                "kernel":
                self.kernel,
                "gamma":
                "auto",
                "random_state":
                self.kernel_seed_tape[trial.number]
            }
        svm = SVR(**parms)
        return svm


# XGboost
class XGBoost_tuner(Basic_tuner):
    """
    Tuning a XGBoost classifier model.    
    [xgboost.XGBClassifier](https://xgboost.readthedocs.io/en/stable/python/python_api.html)

    ToDo:    
        1. sample imbalance. (Done)
        2. early stop. (give up)
        3. efficiency (optuna.integration.XGBoostPruningCallback).

    """

    def __init__(self,
                 n_try=200,
                 target="neg_mean_squared_error",
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=71,
                 validate_penalty=True):
        """

        Args:
            n_try (int, optional): Times to try. Defaults to 50.
            target (str, optional): The target of hyperparameter tuning. It will pass to sklearn.metrics.get_scorer . Using sklearn.metrics.get_scorer_names() to list available metrics. Defaults to "neg_mean_squared_error" (mse score).    
            kernel_seed (int): random seed for model kernel. 
            valid_seed (int): random seed for cross validation
            optuna_seed (int, optional): random seed for optuna. Defaults to 71.     
        
        """
        super().__init__(n_try=n_try,
                         target=target,
                         kernel_seed=kernel_seed,
                         valid_seed=valid_seed,
                         optuna_seed=optuna_seed,
                         validate_penalty=validate_penalty)

    def create_model(self, trial, default=False):
        if default:
            parms = {
                "n_jobs": None,
                "random_state": self.kernel_seed,
                "verbosity": 0
            }
        else:
            if trial.suggest_categorical("use_subsample", [True, False]):
                sampling_rate = trial.suggest_float('subsample',
                                                    0.5,
                                                    0.9,
                                                    log=True)
                col_sampling_rate = trial.suggest_float(
                    'colsample_bytree', 0.1, 0.9)
            else:
                sampling_rate = 1.
                col_sampling_rate = 1.
            parms = {
                "n_estimators":
                trial.suggest_int('n_estimators', 16, 512, log=True),
                "max_depth":
                trial.suggest_int('max_depth', 2, 16, log=True),
                "gamma":
                trial.suggest_float('gamma', 5e-2, 2e+1, log=True),
                "learning_rate":
                trial.suggest_float('learning_rate', 5e-2, 5e-1, log=True),
                "subsample":
                sampling_rate,
                "colsample_bytree":
                col_sampling_rate,
                #"min_child_weight":
                #trial.suggest_float('min_child_weight', 1e-3, 1e+2, log=True),
                "reg_lambda":
                trial.suggest_float('reg_lambda', 1e-2, 1e+1, log=True),
                "n_jobs":
                None,
                "random_state":
                self.kernel_seed_tape[trial.number],
                "verbosity":
                0,
            }

        xgb = XGBRegressor(**parms)
        return xgb


# lightGBM
class LighGBM_tuner(Basic_tuner):
    """
    Tuning a LighGBM classifier model.    
    [lightgbm.LGBMClassifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)     

    ToDo:    
        1. compare with optuna.integration.lightgbm.LightGBMTuner    
    """

    def __init__(self,
                 n_try=200,
                 target="neg_mean_squared_error",
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=71,
                 validate_penalty=True):
        """

        Args:
            n_try (int, optional): Times to try. Defaults to 50.
            target (str, optional): The target of hyperparameter tuning. It will pass to sklearn.metrics.get_scorer . Using sklearn.metrics.get_scorer_names() to list available metrics. Defaults to "neg_mean_squared_error" (mse score).    
            kernel_seed (int): random seed for model kernel. 
            valid_seed (int): random seed for cross validation
            optuna_seed (int, optional): random seed for optuna. Defaults to 71.     
        
        """
        super().__init__(n_try=n_try,
                         target=target,
                         kernel_seed=kernel_seed,
                         valid_seed=valid_seed,
                         optuna_seed=optuna_seed,
                         validate_penalty=validate_penalty)

    def create_model(self, trial, default=False):
        if default:
            parms = {
                "n_jobs": None,
                "random_state": self.kernel_seed,
                "verbosity": -1
            }
        else:
            depth = trial.suggest_float('max_depth', 3, 16, log=True)
            leaves = trial.suggest_float('num_leaves',
                                         depth * 2 / 3,
                                         depth,
                                         log=True)
            depth = int(np.rint(depth))
            leaves = int(np.floor(np.power(2, leaves)))

            if trial.suggest_categorical("use_subsample", [True, False]):
                sampling_rate = trial.suggest_float('subsample',
                                                    0.5,
                                                    0.9,
                                                    log=True)
                col_sampling_rate = trial.suggest_float(
                    'colsample_bytree', 0.1, 0.9)
            else:
                sampling_rate = 1.
                col_sampling_rate = 1.

            parms = {
                "n_estimators":
                trial.suggest_int('n_estimators', 16, 256, log=True),
                "max_depth":
                depth,
                "num_leaves":
                leaves,
                "learning_rate":
                trial.suggest_float('learning_rate', 1e-2, 1, log=True),
                "subsample_freq":
                1,
                "subsample":
                sampling_rate,
                "colsample_bytree":
                col_sampling_rate,
                "min_child_samples":
                trial.suggest_int('min_child_samples', 2, 32, log=True),
                "reg_lambda":
                trial.suggest_float('reg_lambda', 5e-3, 1e+1, log=True),
                "n_jobs":
                None,
                "class_weight":
                "balanced",
                "random_state":
                self.kernel_seed_tape[trial.number],
                "verbosity":
                -1,
            }

        lgbm = LGBMRegressor(**parms)
        return lgbm


# AdaBoost
class AdaBoost_tuner(Basic_tuner):
    """
    Tuning a AdaBoost regressor.    
    [sklearn.ensemble.AdaBoostRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html)
    """

    def __init__(self,
                 n_try=50,
                 target="neg_mean_squared_error",
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=71,
                 validate_penalty=True):
        """
        Args:
            n_try (int, optional): Times to try. Defaults to 50.
            target (str, optional): The target of hyperparameter tuning. It will pass to sklearn.metrics.get_scorer . Using sklearn.metrics.get_scorer_names() to list available metrics. Defaults to "neg_mean_squared_error" (negative mse).    
            kernel_seed (int): random seed for model kernel. 
            valid_seed (int): random seed for cross validation
            optuna_seed (int, optional): random seed for optuna. Defaults to 71.     
        """
        super().__init__(n_try=n_try,
                         target=target,
                         kernel_seed=kernel_seed,
                         valid_seed=valid_seed,
                         optuna_seed=optuna_seed,
                         validate_penalty=validate_penalty)

    def create_model(self, trial, default=False):
        if default:
            parms = {
                "random_state": self.kernel_seed,
            }
        else:
            # scaling penalty: https://scikit-learn.org/stable/auto_examples/svm/plot_svm_scale_c.html#sphx-glr-auto-examples-svm-plot-svm-scale-c-py
            parms = {
                "n_estimators":
                trial.suggest_int('n_estimators', 8, 256, log=True),
                "learning_rate":
                trial.suggest_float('learning_rate', 1e-2, 1, log=True),
                "loss":
                trial.suggest_categorical("loss",
                                          ["linear", "square", "exponential"]),
                "random_state":
                self.kernel_seed_tape[trial.number]
            }
        ada = AdaBoostRegressor(**parms)
        return ada


# DT
class DecisionTree_tuner(Basic_tuner):
    """
    Tuning a DecisionTree regressor.    
    [sklearn.tree.DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
    """

    def __init__(self,
                 n_try=25,
                 target="neg_mean_squared_error",
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=71,
                 validate_penalty=True):
        """
        Args:
            n_try (int, optional): Times to try. Defaults to 50.
            target (str, optional): The target of hyperparameter tuning. It will pass to sklearn.metrics.get_scorer . Using sklearn.metrics.get_scorer_names() to list available metrics. Defaults to "neg_mean_squared_error" (negative mse).    
            kernel_seed (int): random seed for model kernel. 
            valid_seed (int): random seed for cross validation
            optuna_seed (int, optional): random seed for optuna. Defaults to 71.     
        """
        super().__init__(n_try=n_try,
                         target=target,
                         kernel_seed=kernel_seed,
                         valid_seed=valid_seed,
                         optuna_seed=optuna_seed,
                         validate_penalty=validate_penalty)

    def create_model(self, trial, default=False):
        if default:
            parms = {
                "random_state": self.kernel_seed,
            }
        else:
            parms = {
                "max_depth":
                trial.suggest_int('max_depth', 2, 16, log=True),
                "min_samples_split":
                trial.suggest_int('min_samples_split', 2, 32, log=True),
                "min_samples_leaf":
                trial.suggest_int('min_samples_leaf', 1, 16, log=True),
                "ccp_alpha":
                trial.suggest_float('ccp_alpha', 1e-3, 1e-1, log=True),
                "random_state":
                self.kernel_seed_tape[trial.number],
                "class_weight":
                "balanced"
            }
        DT = DecisionTreeRegressor(**parms)
        return DT


# CatBoost
# KNN
# KNN-Graph spectrum
