from . import Basic_tuner

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import numpy as np
from statsmodels.discrete.discrete_model import Logit

# ToDo: multi class support
# ToDo: optuna pruner
#       see section Acticating Pruners in https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html


# linear model
class ElasticLogit_tuner(Basic_tuner):
    """
    Tuning a elasic net logistic regression.    
    [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression), reminds the choice of the algorithm depends on the penalty chosen and on (multinomial) multiclass support.    
    """

    def __init__(self,
                 kernel="saga",
                 n_try=25,
                 target="matthews_corrcoef",
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=71):
        """
        Args:
            kernel (str, optional): It will be passed to "solver" of sklearn.linear_model.LogisticRegression . Defaults to "saga".    
            n_try (int, optional): Times to try. Defaults to 25.    
            target (str, optional): The target of hyperparameter tuning. It will pass to sklearn.metrics.get_scorer . Using sklearn.metrics.get_scorer_names() to list available metrics. Defaults to "matthews_corrcoef" (mcc score).    
            kernel_seed (int): random seed for model kernel. 
            valid_seed (int): random seed for cross validation
            optuna_seed (int, optional): random seed for optuna. Defaults to 71.     
        """
        super().__init__(n_try=n_try,
                         target=target,
                         kernel_seed=kernel_seed,
                         valid_seed=valid_seed,
                         optuna_seed=optuna_seed)

        # ‘saga’ fast convergence is only guaranteed on features with approximately the same scale. You should do a feature-wise (between sample) normalization before fitting.
        self.kernel = kernel  # sage, lbfgs
        self.penalty = "elasticnet"
        #print("Logistic Regression with: ", self.penalty, " penalty, ", self.kernel, " solver.")

    def create_model(self, trial, default=False):
        if default:
            parms = {
                "verbose": 0,
                "l1_ratio": 0.5,
                "penalty": self.penalty,
                "solver": self.kernel
            }
        else:
            parms = {
                "C": trial.suggest_float('C', 1e-6, 1e+2, log=True),
                "l1_ratio": trial.suggest_float('l1_ratio', 0, 1),
                "penalty": self.penalty,
                "class_weight": "balanced",
                "solver": self.kernel,
                "verbose": 0,
            }
        lg = LogisticRegression(**parms)
        return lg

    def summary(self):
        """
        It is the way I found to cram a sklearn regression result into the statsmodel regresion.    
        The only reason to do this is that statsmodel provides R-style summary.    
        """
        sm_logit = Logit(self.y, self.x).fit(
            disp=False,
            start_params=self.best_model.coef_.flatten(),
            maxiter=0,
            warn_convergence=False)
        print(sm_logit.summary())


# RF
class RandomForest_tuner(Basic_tuner):
    """
    Tuning a random forest model.    
    [sklearn.ensemble.RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
    """

    def __init__(self,
                 using_oob=True,
                 n_try=50,
                 target="matthews_corrcoef",
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=71):
        """

        Args:
            using_oob (bool, optional): Using out of bag score as validation. Defaults to True.
            n_try (int, optional): Times to try. Defaults to 50.
            target (str, optional): The target of hyperparameter tuning. It will pass to sklearn.metrics.get_scorer . Using sklearn.metrics.get_scorer_names() to list available metrics. Defaults to "neg_mean_squared_error" (mcc score).    
            kernel_seed (int): random seed for model kernel. 
            valid_seed (int): random seed for cross validation
            optuna_seed (int, optional): random seed for optuna. Defaults to 71.     
        """
        super().__init__(n_try=n_try,
                         target=target,
                         kernel_seed=kernel_seed,
                         valid_seed=valid_seed,
                         optuna_seed=optuna_seed)

        self.using_oob = using_oob

    def create_model(self, trial, default=False):
        if default:
            parms = {
                "bootstrap": self.using_oob,
                "oob_score": self.using_oob,
                "n_jobs": -1,
                "random_state": self.kernel_seed,
                "verbose": 0,
            }
        else:
            parms = {
                "n_estimators":
                trial.suggest_int('n_estimators', 32, 1024, log=True),
                "max_depth":
                trial.suggest_int('max_depth', 2, 16, log=True),
                "min_samples_split":
                trial.suggest_int('min_samples_split', 2, 16, log=True),
                "min_samples_leaf":
                trial.suggest_int('min_samples_leaf', 1, 16, log=True),
                "ccp_alpha":
                trial.suggest_float('ccp_alpha', 1e-3, 1e-1, log=True),
                "max_samples":
                trial.suggest_float('max_samples', 0.5, 0.95, log=True),
                "bootstrap":
                self.using_oob,
                "oob_score":
                self.using_oob,
                "n_jobs":
                -1,
                "random_state":
                self.kernel_seed_tape[trial.number],
                "verbose":
                0,
                "class_weight":
                "balanced"
            }

        rf = RandomForestClassifier(**parms)
        return rf

    def evaluate(self, trial, default=False):
        """
        RF needs oob and we have it.
        
        Args:
            trial (optuna.trial.Trial): optuna trial in this call.
            default (bool): To use default hyper parameter. This argument will be passed to creat_model
        Returns :
            float: The score.
        """
        classifier_obj = self.create_model(trial, default)

        if self.using_oob:
            # oob predict
            classifier_obj.fit(self.x, self.y)
            y_pred = classifier_obj.oob_decision_function_[:, 1]

            # oob score
            if not self.eval_prob:
                # !!! some metrics (such as accuracy and F1) require discrete predictions.
                # !!! If the score function raise an error about y_pred should not be float,
                # !!! then please add the name of your metric(target) into Basic_tuner's "discrete_target" in ./__init__.py.
                y_pred = y_pred > 0.5
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
                scoring=self.metric)
            score = score.mean()
        return score


# SVM
class SVM_tuner(Basic_tuner):
    """
    Tuning a support vector machine.    
    [sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
    """

    def __init__(self,
                 kernel="rbf",
                 n_try=25,
                 target="matthews_corrcoef",
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=71):
        """
        Args:
            kernel (str, optional): This will be passed to the attribute of SVC: "kernel". Defaults to "rbf".
            n_try (int, optional): Times to try. Defaults to 50.
            target (str, optional): The target of hyperparameter tuning. It will pass to sklearn.metrics.get_scorer . Using sklearn.metrics.get_scorer_names() to list available metrics. Defaults to "neg_mean_squared_error" (mcc score).    
            kernel_seed (int): random seed for model kernel. 
            valid_seed (int): random seed for cross validation
            optuna_seed (int, optional): random seed for optuna. Defaults to 71.     
        """
        super().__init__(n_try=n_try,
                         target=target,
                         kernel_seed=kernel_seed,
                         valid_seed=valid_seed,
                         optuna_seed=optuna_seed)
        self.kernel = kernel  # rbf, linear, poly, sigmoid

    def create_model(self, trial, default=False):
        if default:
            parms = {
                "kernel": self.kernel,
                "random_state": self.kernel_seed,
                "probability": True
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
                "class_weight":
                "balanced",
                "gamma":
                "auto",
                "probability":
                True,
                "random_state":
                self.kernel_seed_tape[trial.number]
            }
        svm = SVC(**parms)
        return svm


# XGboost
class XGBoost_tuner(Basic_tuner):
    """
    Tuning a XGBoost classifier model.    
    [xgboost.XGBClassifier](https://xgboost.readthedocs.io/en/stable/python/python_api.html)

    ToDo:    
        1. sample imbalance. (we have temporary solution)    
        2. early stop.    
        3. efficiency (optuna.integration.XGBoostPruningCallback).    

    """

    def __init__(self,
                 n_try=100,
                 target="matthews_corrcoef",
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=71):
        """

        Args:
            n_try (int, optional): Times to try. Defaults to 50.
            target (str, optional): The target of hyperparameter tuning. It will pass to sklearn.metrics.get_scorer . Using sklearn.metrics.get_scorer_names() to list available metrics. Defaults to "neg_mean_squared_error" (mcc score).    
            kernel_seed (int): random seed for model kernel. 
            valid_seed (int): random seed for cross validation
            optuna_seed (int, optional): random seed for optuna. Defaults to 71.     
        
        """
        super().__init__(n_try=n_try,
                         target=target,
                         kernel_seed=kernel_seed,
                         valid_seed=valid_seed,
                         optuna_seed=optuna_seed)

    def create_model(self, trial, default=False):
        if default:
            parms = {
                "n_jobs": None,
                "random_state": self.kernel_seed,
                "verbosity": 0
            }
        else:
            parms = {
                "n_estimators":
                trial.suggest_int('n_estimators', 16, 256, log=True),
                "max_depth":
                trial.suggest_int('max_depth', 2, 16, log=True),
                "gamma":
                trial.suggest_float('gamma', 5e-2, 2e+1, log=True),
                "learning_rate":
                trial.suggest_float('learning_rate', 5e-3, 5e-1, log=True),
                "subsample":
                trial.suggest_float('subsample', 0.5, 0.95, log=True),
                "colsample_bytree":
                trial.suggest_float('colsample_bytree', 0.8, 1),
                "min_child_weight":
                trial.suggest_float('min_child_weight', 1e-2, 1e+2, log=True),
                "reg_lambda":
                trial.suggest_float('reg_lambda', 1e-2, 1e+1, log=True),
                "n_jobs":
                None,
                "random_state":
                self.kernel_seed_tape[trial.number],
                "verbosity":
                0,
            }

        xgb = XGBClassifier(**parms)
        return xgb

    def evaluate(self, trial, default=False):
        """
        xgboost do not support class_weight.    
        we use sample_weight as substitute.    

        Args:
            trial (optuna.trial.Trial): optuna trial in this call.
            default (bool): To use default hyper parameter. This argument will be passed to creat_model
        Returns :
            float: The score.
        """
        classifier_obj = self.create_model(trial, default)

        cv = StratifiedKFold(n_splits=5,
                             shuffle=True,
                             random_state=self.valid_seed_tape[trial.number])

        score = []
        for i, (train_ind, test_ind) in enumerate(cv.split(self.x, self.y)):
            x_train = self.x.iloc[train_ind]
            y_train = self.y.iloc[train_ind]
            x_test = self.x.iloc[test_ind]
            y_test = self.y.iloc[test_ind]

            classifier_obj.fit(x_train,
                               y_train,
                               sample_weight=compute_sample_weight(
                                   class_weight="balanced", y=y_train))

            score.append(self.metric(classifier_obj, x_test, y_test))
        score = np.array(score).mean()
        return score


# lightGBM
class LighGBM_tuner(Basic_tuner):
    """
    Tuning a LighGBM classifier model.    
    [lightgbm.LGBMClassifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)     

    ToDo:    
        1. compare with optuna.integration.lightgbm.LightGBMTuner    
    """

    def __init__(self,
                 n_try=100,
                 target="matthews_corrcoef",
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=71):
        """

        Args:
            n_try (int, optional): Times to try. Defaults to 50.
            target (str, optional): The target of hyperparameter tuning. It will pass to sklearn.metrics.get_scorer . Using sklearn.metrics.get_scorer_names() to list available metrics. Defaults to "neg_mean_squared_error" (mcc score).    
            kernel_seed (int): random seed for model kernel. 
            valid_seed (int): random seed for cross validation
            optuna_seed (int, optional): random seed for optuna. Defaults to 71.     
        
        """
        super().__init__(n_try=n_try,
                         target=target,
                         kernel_seed=kernel_seed,
                         valid_seed=valid_seed,
                         optuna_seed=optuna_seed)

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
            parms = {
                "n_estimators":
                trial.suggest_int('n_estimators', 16, 256, log=True),
                "max_depth":
                depth,
                "num_leaves":
                leaves,
                "learning_rate":
                trial.suggest_float('learning_rate', 1e-2, 1, log=True),
                "subsample":
                trial.suggest_float('subsample', 0.5, 0.95, log=True),
                "subsample_freq":
                1,
                "colsample_bytree":
                trial.suggest_float('colsample_bytree', 0.7, 1, log=True),
                "min_child_weight":
                trial.suggest_float('min_child_weight', 1e-5, 1e-1, log=True),
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

        lgbm = LGBMClassifier(**parms)
        return lgbm


# Todo
# KNN
# KNN-Graph spectrum
