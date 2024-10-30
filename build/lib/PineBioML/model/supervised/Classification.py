from . import Basic_tuner
from abc import abstractmethod
from typing import Literal

from joblib import parallel_backend

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from statsmodels.discrete.discrete_model import Logit
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool

import numpy as np
from pandas import DataFrame


class Classification_tuner(Basic_tuner):
    """
    A subclass of Basic_tuner for classification task.

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
        No, this is not a regresion tuner or for regresion task.
        Returns:
            bool: False
        """

        return False

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

    def predict_proba(self, x):
        """
        The sklearn.base.BaseEstimator predict_prob api.

        Args:
            x (pandas.DataFrame): feature to extract information from.

        Returns:
            pd.DataFrame: prediction in prob, an array with shape (n_samples, n_classes)
        """
        return DataFrame(self.best_model.predict_proba(x),
                         index=x.index,
                         columns=self.y_mapping.classes_)


# linear model, elasticnet
class ElasticLogit_tuner(Classification_tuner):
    """
    Tuning a elasic net logistic regression.    
    [sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression), reminds the choice of the algorithm depends on the penalty chosen and on (multinomial) multiclass support.    
    """

    def __init__(self,
                 n_try=25,
                 n_cv=5,
                 target="mcc",
                 kernel_seed: int = None,
                 valid_seed: int = None,
                 optuna_seed: int = None,
                 validate_penalty=True):
        """

        Args:
            n_try (int, optional): The number of trials optuna should try. Defaults to 25.
            n_cv (int, optional): The number of folds to execute cross validation evaluation in iteration of optuna optimization. Defaults to 5.
            target (str, optional): The target of optuna optimization. Notice that is different from the training loss of model. Defaults to "mcc".
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

        # "saga" fast convergence is only guaranteed on features with approximately the same scale. You should do a feature-wise (between sample) normalization before fitting.
        self.kernel = "saga"
        self.penalty = "elasticnet"

    def name(self):
        return "ElasticNetLogisticRegression"

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
            " document"] = "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"
        return refer

    def create_model(self, trial, default=False):
        if default:
            parms = {
                "C": 1.0,
                "l1_ratio": 0.5,
                "penalty": self.penalty,
                "solver": self.kernel,
                "random_state": self.kernel_seed_tape[trial.number],
                "verbose": 0
            }
        else:
            parms = {
                "C": trial.suggest_float('C', 1e-6, 1e+2, log=True),
                "l1_ratio": trial.suggest_float('l1_ratio', 0, 1),
                "penalty": self.penalty,
                "solver": self.kernel,
                "random_state": self.kernel_seed_tape[trial.number],
                "verbose": 0,
            }
        lg = LogisticRegression(**parms)
        return lg

    def summary(self):
        """
        It is the way I found to cram a sklearn regression result into the statsmodel regresion.    
        The only reason to do this is that statsmodel provides R-style summary.    
        """
        if len(self.best_model.classes_) > 2:
            # multi-class classification
            # Todo
            raise TypeError(
                "multi-class classification summary not support yet. Please tell me why do you need that in a multi-class classification task"
            )
        else:
            # binary classification
            sm_logit = Logit(self.y, self.x).fit(
                disp=False,
                start_params=self.best_model.coef_.flatten(),
                maxiter=0,
                warn_convergence=False)
        print(sm_logit.summary())


# RF
class RandomForest_tuner(Classification_tuner):
    """
    Tuning a random forest model.    
    [sklearn.ensemble.RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
    """

    def __init__(self,
                 using_oob=True,
                 n_try=50,
                 n_cv=5,
                 target="mcc",
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=None,
                 validate_penalty=True):
        """

        Args:
            using_oob (bool, optional): Using out of bag score as validation. Defaults to True.
            n_try (int, optional): The number of trials optuna should try. Defaults to 50.
            n_cv (int, optional): The number of folds to execute cross validation evaluation in iteration of optuna optimization. Defaults to 5.
            target (str, optional): The target of optuna optimization. Notice that is different from the training loss of model. Defaults to "mcc".
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
            " document"] = "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"
        refer[
            self.name() +
            " publication"] = "https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf"

        return refer

    def create_model(self, trial, default=False):
        if default:
            parms = {
                "bootstrap": self.using_oob,
                "oob_score": self.using_oob,
                "n_jobs": -1,
                "random_state": self.kernel_seed_tape[trial.number],
                "verbose": 0,
            }
        else:
            parms = {
                "n_estimators":
                trial.suggest_int('n_estimators', 32, 1024, log=True),
                "min_samples_leaf":
                trial.suggest_int('min_samples_leaf', 1, 32, log=True),
                "ccp_alpha":
                trial.suggest_float('ccp_alpha', 1e-4, 1e-1, log=True),
                "max_samples":
                trial.suggest_float('max_samples', 0.5, 0.9, log=True),
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
            # there is a bug that default sklaern randomforest parallel_backend using thread where others use "loky", see joblib.parallel_backend.
            with parallel_backend('loky'):
                classifier_obj.fit(self.x,
                                   self.y,
                                   sample_weight=compute_sample_weight(
                                       class_weight="balanced", y=self.y))

            # oob prediction
            y_pred = classifier_obj.oob_decision_function_
            if self.metric_name == "roc_auc":
                # roc_auc can only be used on binary classification. Do not try ovr, ovo. forget them.
                y_pred = y_pred[:, 1]

            # oob score
            ### manual scorer wraper.
            if self.metric_using_proba:
                score = self.metric._score_func(self.y, y_pred,
                                                **self.scorer_kargs)
            else:
                # revert to class symbols.
                y_pred = classifier_obj.classes_[y_pred.argmax(axis=-1)]
                score = self.metric._score_func(self.y, y_pred,
                                                **self.scorer_kargs)
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
                    classifier_obj.fit(x_train,
                                       y_train,
                                       sample_weight=compute_sample_weight(
                                           class_weight="balanced", y=y_train))

                test_score = self.metric(classifier_obj, x_test, y_test)
                train_score = self.metric(classifier_obj, x_train, y_train)
                if self.validate_penalty:
                    score.append(test_score + 0.1 * (test_score - train_score))
                else:
                    score.append(test_score)

            score = sum(score) / self.n_cv
        return score


# SVM
class SVM_tuner(Classification_tuner):
    """
    Tuning a support vector machine.    
    [sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
    """

    def __init__(self,
                 kernel: Literal["linear", "poly", "rbf", "sigmoid"] = "rbf",
                 n_try=25,
                 n_cv=5,
                 target="mcc",
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=None,
                 validate_penalty=True):
        """

        Args:
            kernel (Literal[&quot;linear&quot;, &quot;poly&quot;, &quot;rbf&quot;, &quot;sigmoid&quot;], optional): This will be passed to the attribute of SVC: "kernel". Defaults to "rbf".
            n_try (int, optional): The number of trials optuna should try. Defaults to 25.
            n_cv (int, optional): The number of folds to execute cross validation evaluation in iteration of optuna optimization. Defaults to 5.
            target (str, optional): The target of optuna optimization. Notice that is different from the training loss of model. Defaults to "mcc".
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
        self.kernel = kernel  # rbf, linear, poly, sigmoid

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
            " document"] = "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html"
        refer[
            self.name() +
            " publication"] = "https://citeseerx.ist.psu.edu/doc_view/pid/42e5ed832d4310ce4378c44d05570439df28a393"

        return refer

    def create_model(self, trial, default=False):
        if default:
            parms = {
                "kernel": self.kernel,
                "random_state": self.kernel_seed_tape[trial.number],
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
                "gamma":
                "auto",
                "kernel":
                self.kernel,
                "random_state":
                self.kernel_seed_tape[trial.number],
                "probability":
                True
            }
        svm = SVC(**parms)
        return svm


# Todo: learning rate and number of iteration adjustment
# XGboost
class XGBoost_tuner(Classification_tuner):
    """
    Tuning a XGBoost classifier model.    
    [xgboost.XGBClassifier](https://xgboost.readthedocs.io/en/stable/python/python_api.html)

    ToDo:    
        1. sample imbalance. (we have temporary solution)    
        2. early stop.    
        3. efficiency (optuna.integration.XGBoostPruningCallback).    

    """

    def __init__(self,
                 n_try=75,
                 n_cv=5,
                 target="mcc",
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=None,
                 validate_penalty=True):
        """

        Args:
            n_try (int, optional): The number of trials optuna should try. Defaults to 75.
            n_cv (int, optional): The number of folds to execute cross validation evaluation in iteration of optuna optimization. Defaults to 5.
            target (str, optional): The target of optuna optimization. Notice that is different from the training loss of model. Defaults to "mcc".
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

    def create_model(self, trial, default=False):
        if default:
            parms = {
                "n_jobs": None,
                "random_state": self.kernel_seed_tape[trial.number],
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
                trial.suggest_int('n_estimators', 16, 256, log=True),
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


# lightGBM
class LighGBM_tuner(Classification_tuner):
    """
    Tuning a LighGBM classifier model.    
    [lightgbm.LGBMClassifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)     

    ToDo:    
        1. compare with optuna.integration.lightgbm.LightGBMTuner    
    """

    def __init__(self,
                 n_try=75,
                 n_cv=5,
                 target="mcc",
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=None,
                 validate_penalty=True):
        """

        Args:
            n_try (int, optional): The number of trials optuna should try. Defaults to 75.
            n_cv (int, optional): The number of folds to execute cross validation evaluation in iteration of optuna optimization. Defaults to 5.
            target (str, optional): The target of optuna optimization. Notice that is different from the training loss of model. Defaults to "mcc".
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
                "random_state":
                self.kernel_seed_tape[trial.number],
                "verbosity":
                -1,
            }

        lgbm = LGBMClassifier(**parms)
        return lgbm


# Adaboost
class AdaBoost_tuner(Classification_tuner):
    """
    Tuning a AdaBoost calssifier.    
    [sklearn.ensemble.AdaBoostClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
    """

    def __init__(self,
                 n_try=25,
                 n_cv=5,
                 target="mcc",
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=None,
                 validate_penalty=True):
        """
        Args:
            n_try (int, optional): The number of trials optuna should try. Defaults to 25.
            n_cv (int, optional): The number of folds to execute cross validation evaluation in iteration of optuna optimization. Defaults to 5.
            target (str, optional): The target of optuna optimization. Notice that is different from the training loss of model. Defaults to "mcc".
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
            " document"] = "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html"
        refer[
            self.name() +
            " publication: original version"] = "https://www.ee.columbia.edu/~sfchang/course/svia-F03/papers/freund95decisiontheoretic-adaboost.pdf"
        refer[
            self.name() +
            " publication: implemented version"] = "https://doi.org/10.4310/SII.2009.v2.n3.a8"

        return refer

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
                "algorithm":
                "SAMME",
                "random_state":
                self.kernel_seed_tape[trial.number]
            }
        ada = AdaBoostClassifier(**parms)
        return ada


# DT
class DecisionTree_tuner(Classification_tuner):
    """
    Tuning a DecisionTree calssifier.    
    [sklearn.tree.DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
    """

    def __init__(self,
                 n_try=25,
                 n_cv=5,
                 target="mcc",
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=None,
                 validate_penalty=True):
        """
        Args:
            n_try (int, optional): The number of trials optuna should try. Defaults to 25.
            n_cv (int, optional): The number of folds to execute cross validation evaluation in iteration of optuna optimization. Defaults to 5.
            target (str, optional): The target of optuna optimization. Notice that is different from the training loss of model. Defaults to "mcc".
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
            " document"] = "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html"
        refer[
            self.name() +
            " publication"] = "https://www.taylorfrancis.com/books/mono/10.1201/9781315139470/classification-regression-trees-leo-breiman-jerome-friedman-olshen-charles-stone"

        return refer

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
            }
        DT = DecisionTreeClassifier(**parms)
        return DT


# catboost
class CatBoost_tuner(Classification_tuner):
    """
    Tuning a CatBoost classifier model.    
    [catboost.CatBoostClassifier](https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier)     

    ToDo:    
        1. compare with optuna.integration.CatBoost...    
    """

    def __init__(self,
                 n_try=75,
                 n_cv=5,
                 target="mcc",
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=None,
                 validate_penalty=True):
        """

        Args:
            n_try (int, optional): The number of trials optuna should try. Defaults to 75.
            n_cv (int, optional): The number of folds to execute cross validation evaluation in iteration of optuna optimization. Defaults to 5.
            target (str, optional): The target of optuna optimization. Notice that is different from the training loss of model. Defaults to "mcc".
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
        return "CatBoost"

    def reference(self) -> dict[str, str]:
        """
        This function will return reference of this method in python dict.    
        If you want to access it in PineBioML api document, then click on the    >Expand source code     

        Returns:
            dict[str, str]: a dict of reference.
        """
        refer = super().reference()
        refer[self.name() + " document"] = "https://catboost.ai/en/docs/"
        refer[
            self.name() +
            " publication"] = "https://proceedings.neurips.cc/paper_files/paper/2018/file/14491b756b3a51daac41c24863285549-Paper.pdf"

        return refer

    def create_model(self, trial, default=False):
        if default:
            parms = {
                "random_seed": self.kernel_seed,
                "verbose": False,
                #"use_best_model": True
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
                "learning_rate":
                trial.suggest_float('learning_rate', 1e-2, 1, log=True),
                "max_depth":
                depth,
                "reg_lambda":
                trial.suggest_float('reg_lambda', 5e-3, 1e+1, log=True),
                "colsample_bylevel":
                col_sampling_rate,
                "subsample":
                sampling_rate,
                #"use_best_model": True,
                "random_seed":
                self.kernel_seed_tape[trial.number],
                "verbose":
                False,
            }

        cat = CatBoostClassifier(**parms)
        return cat

    def evaluate(self, trial, default=False):
        """
        To evaluate the score of this trial. you should call create_model instead of creating model manually in this function.    
        catboost need to be used with pool.
        
        Args:
            trial (optuna.trial.Trial): optuna trial in this call.
            default (bool): To use default hyper parameter. This argument will be passed to creat_model
        Returns :
            float: The score.
        """
        classifier_obj = self.create_model(trial, default)

        cv = StratifiedKFold(n_splits=self.n_cv,
                             shuffle=True,
                             random_state=self.valid_seed_tape[trial.number])

        score = []
        for i, (train_ind, test_ind) in enumerate(cv.split(self.x, self.y)):
            x_train = self.x.iloc[train_ind]
            y_train = self.y.iloc[train_ind]
            pool_train = Pool(x_train, y_train)

            x_test = self.x.iloc[test_ind]
            y_test = self.y.iloc[test_ind]
            #pool_test = Pool(x_test, y_test)

            classifier_obj.fit(pool_train)  #, eval_set=pool_test

            train_score = self.metric(classifier_obj, x_train, y_train)
            test_score = self.metric(classifier_obj, x_test, y_test)
            if self.validate_penalty:
                score.append(test_score + 0.1 * (test_score - train_score))
            else:
                score.append(test_score)

        score = sum(score) / self.n_cv
        return score


# Todo
# KNN
# KNN-Graph spectrum
