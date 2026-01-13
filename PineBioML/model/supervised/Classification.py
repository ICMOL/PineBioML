from . import Basic_tuner
from joblib import parallel_config

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from statsmodels.discrete.discrete_model import Logit
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping
from catboost import CatBoostClassifier, Pool

import shap

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
        No, this is not a regresion tuner or for regresion task.
        Returns:
            bool: False
        """

        return False

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
                 validate_penalty=True,
                 TT_coef=0.1,
                 kernel_seed: int = None,
                 valid_seed: int = None,
                 optuna_seed: int = None):

        super().__init__(n_try=n_try,
                         n_cv=n_cv,
                         target=target,
                         validate_penalty=validate_penalty,
                         TT_coef=TT_coef,
                         kernel_seed=kernel_seed,
                         valid_seed=valid_seed,
                         optuna_seed=optuna_seed)

        # "saga" fast convergence is only guaranteed on features with approximately the same scale. You should do a feature-wise (between sample) normalization before fitting.
        self.solver = "saga"
        self.n_opt_jobs = 4

    def name(self):
        return "ElasticNetLogisticRegression"

    def reference(self) -> dict[str, str]:
        refer = super().reference()
        refer[
            self.name() +
            " document"] = "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html"
        return refer

    def parms_range(self) -> dict:
        return {
            "C": ('C', "float", 1e-6, 1e+2),
            "l1_ratio": ('l1_ratio', "float", -0.5, 1.5),
        }

    def create_model(self, trial, default=False, training=False):
        parms = {
            "C": 1.0,
            "l1_ratio": 0.,
            "penalty": "elasticnet",
            "solver": "saga",
            "random_state": self.kernel_seed,
            "verbose": 0
        }
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
        lg = LogisticRegression(**parms)
        return lg

    def summary(self):
        """
        It is the way I found to cram a sklearn regression result into the statsmodel regresion.    
        The only reason to do this is that statsmodel provides R-style summary.    
        """
        if len(self.best_model.classes_) > 2:
            # Todo: multi-class classification
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

    def _explainer(self, x):
        return shap.LinearExplainer(self.best_model, x)


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
            " document"] = "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"
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
        }

        if not default:
            parms_to_tune = self.parms_range()
            for par in parms_to_tune:
                parms[par] = self.parms_range_sparser(trial,
                                                      parms_to_tune[par])
            parms["random_state"] = self.kernel_seed_tape[trial.number]

        rf = RandomForestClassifier(**parms)
        return rf

    def evaluate(self, trial, default=None, training=None):
        """
        RF has oob validation and we have it.
        
        """
        if default is None:
            default = self.default
        if training is None:
            training = self.training
        classifier_obj = self.create_model(trial, default)

        if self.using_oob:
            # oob predict
            # there is a bug that default sklaern randomforest parallel_backend using thread where others use "loky", see joblib.parallel_backend.
            with parallel_config(backend='loky'):
                classifier_obj.fit(self.x,
                                   self.y,
                                   sample_weight=compute_sample_weight(
                                       class_weight="balanced", y=self.y))

            # oob prediction
            y_prob = classifier_obj.oob_decision_function_

            # tune a threshold via roc for Binary classification
            """
            if self.is_binary:
                if training:
                    fpr, tpr, thr = roc_curve(self.y, y_prob[:, 1])
                    self.thresholds[trial.number] = thr[abs(tpr -
                                                            fpr).argmax()]
            """

            # oob score
            ### manual scorer wraper.
            if self.metric_using_proba:
                if self.metric_name == "roc_auc":
                    # roc_auc can only be used on binary classification. Do not try ovr, ovo. forget them.
                    y_prob = y_prob[:, 1]
                score = self.metric._score_func(self.y, y_prob,
                                                **self.scorer_kargs)
            else:
                # revert to class symbols.
                """
                if self.is_binary:
                    if default:
                        t = 0.5
                    else:
                        t = self.thresholds[trial.number]
                    y_pred = classifier_obj.classes_[(y_prob[:, 1]
                                                      > t).astype(np.int16)]
                else:
                """
                y_pred = classifier_obj.classes_[y_prob.argmax(axis=-1)]
                score = self.metric._score_func(self.y, y_pred,
                                                **self.scorer_kargs)
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
class SVM_tuner(Classification_tuner):
    """
    Tuning a support vector machine.    
    [sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
    """

    def __init__(self,
                 n_try=25,
                 n_cv=5,
                 target="mcc",
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
            "random_state": self.kernel_seed,
            "probability": True,
            "gamma": "auto"
        }
        if not default:
            parms_to_tune = self.parms_range()
            for par in parms_to_tune:
                parms[par] = self.parms_range_sparser(trial,
                                                      parms_to_tune[par])
            #parms["random_state"] = self.kernel_seed_tape[trial.number]
        svm = SVC(**parms)
        return svm

    def _explainer(self, x):
        return shap.KernelExplainer(self.best_model.predict_proba, x)


# Todo: learning rate and number of iteration adjustment
# XGboost
class XGBoost_tuner(Classification_tuner):
    """
    Tuning a XGBoost classifier model.    
    [xgboost.XGBClassifier](https://xgboost.readthedocs.io/en/stable/python/python_api.html)

    ToDo:    
        1. sample imbalance. (we have temporary solution)    

    """

    def __init__(self,
                 n_try=75,
                 n_cv=5,
                 target="mcc",
                 validate_penalty=True,
                 TT_coef=0.1,
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=None):
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
                         validate_penalty=validate_penalty,
                         TT_coef=TT_coef,
                         kernel_seed=kernel_seed,
                         valid_seed=valid_seed,
                         optuna_seed=optuna_seed)

    def name(self):
        return "XGBoost"

    def reference(self) -> dict[str, str]:

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
            "subsample": ('subsample', "float", 0.5, 1),
            "colsample_bytree": ('colsample_bytree', "float", 0.5, 1),
            "reg_lambda": ('reg_lambda', "float", 1e-3, 1),
            "reg_alpha": ('reg_alpha', "float", 1e-4, 1.)
        }

    def create_model(self, trial, default=False, training=False):
        parms = {
            "n_jobs": None,
            "random_state": self.kernel_seed,
            "verbosity": 0,
            "enable_categorical": True
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

        xgb = XGBClassifier(**parms)
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
class LightGBM_tuner(Classification_tuner):
    """
    Tuning a LighGBM classifier model.    
    [lightgbm.LGBMClassifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)     

    ToDo:    
        1. compare with optuna.integration.lightgbm.LightGBMTuner    
    """

    def __init__(self,
                 n_try=50,
                 n_cv=5,
                 target="mcc",
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
            ("min_child_samples", "int", 1, round(np.sqrt(self.n_sample) / 2)),
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

        lgbm = LGBMClassifier(**parms)
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
        self.n_opt_jobs = 4

    def name(self):
        return "AdaBoost"

    def reference(self) -> dict[str, str]:
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

    def parms_range(self) -> dict:
        return {
            "n_estimators": ('n_estimators', "int", 4, 64),
            "learning_rate": ('learning_rate', "float", 1e-2, 1.)
        }

    def create_model(self, trial, default=False, training=False):
        parms = {
            "random_state": self.kernel_seed,
        }
        if not default:
            parms_to_tune = self.parms_range()
            for par in parms_to_tune:
                parms[par] = self.parms_range_sparser(trial,
                                                      parms_to_tune[par])
            parms["random_state"] = self.kernel_seed_tape[trial.number]
        ada = AdaBoostClassifier(**parms)
        return ada

    def _explainer(self, x):
        return shap.KernelExplainer(self.best_model.predict_proba, x)


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
        self.n_opt_jobs = 4

    def name(self):
        return "DecisionTree"

    def reference(self) -> dict[str, str]:
        refer = super().reference()
        refer[
            self.name() +
            " document"] = "https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html"
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
        DT = DecisionTreeClassifier(**parms)
        return DT

    def _explainer(self, x):
        return shap.TreeExplainer(self.best_model)


# catboost
class CatBoost_tuner(Classification_tuner):
    """
    Tuning a CatBoost classifier model.    
    [catboost.CatBoostClassifier](https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier)     

    ToDo:    
        1. compare with optuna.integration.CatBoost...    
    """

    def __init__(self,
                 n_try=50,
                 n_cv=5,
                 target="mcc",
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
        return "CatBoost"

    def reference(self) -> dict[str, str]:

        refer = super().reference()
        refer[self.name() + " document"] = "https://catboost.ai/en/docs/"
        refer[
            self.name() +
            " publication"] = "https://proceedings.neurips.cc/paper_files/paper/2018/file/14491b756b3a51daac41c24863285549-Paper.pdf"

        return refer

    def parms_range(self) -> dict:
        return {
            "n_estimators": ('n_estimators', "int", 4, 256),
            "learning_rate": ('learning_rate', "float", 1e-1, 1.),
            "max_depth":
            ("max_depth", "int", round(np.log2(self.n_sample) / 2),
             int(np.log2(self.n_sample)) + 2),
            "reg_lambda": ('reg_lambda', "float", 1e-3, 1.),
            "colsample_bylevel": ('colsample_bytree', "float", 0.5, 1.),
            "subsample": ('subsample', "float", 0.5, 1.)
        }

    def create_model(self, trial, default=False, training=False):
        parms = {
            "random_state": self.kernel_seed,
            "verbose": False,
        }
        if not default:
            parms_to_tune = self.parms_range()
            for par in parms_to_tune:
                parms[par] = self.parms_range_sparser(trial,
                                                      parms_to_tune[par])
            parms["random_state"] = self.kernel_seed_tape[trial.number]
        cat = CatBoostClassifier(**parms)
        return cat

    def optimize_fit(self, clr, train_data, sample_weight, valid_data):
        train_x, train_y = train_data
        valid_x, valid_y = valid_data

        pool_train = Pool(train_x, train_y)
        pool_valid = Pool(valid_x, valid_y)
        cat_features = list(train_x.columns[train_x.dtypes == "category"])

        return clr.fit(pool_train,
                       cat_features=cat_features,
                       sample_weight=sample_weight,
                       eval_set=pool_valid,
                       verbose=False,
                       early_stopping_rounds=round(clr.n_estimators * 0.1) + 2)

    # TODO: integrate to optimize_fit
    def _evaluate(self, trial, default=None, training=None):
        """
        To evaluate the score of this trial. you should call create_model instead of creating model manually in this function.    
        catboost need to be used with pool.
        
        Args:
            trial (optuna.trial.Trial): optuna trial in this call.
            default (bool): To use default hyper parameter. This argument will be passed to creat_model
        Returns :
            float: The score.
        """
        if default is None:
            default = self.default
        if training is None:
            training = self.training
        classifier_obj = self.create_model(trial, default)

        cv = StratifiedKFold(n_splits=self.n_cv,
                             shuffle=True,
                             random_state=self.valid_seed_tape[trial.number])

        # do cv
        score = np.zeros(self.n_cv)
        # thresholds not None only if  self.is_binary and not self.is_regression
        #cv_thresholds = np.zeros(self.n_cv)
        cv_stoppoint = np.zeros(self.n_cv)
        for i, (train_ind, test_ind) in enumerate(cv.split(self.x, self.y)):
            x_train = self.x.iloc[train_ind]
            y_train = self.y.iloc[train_ind]
            pool_train = Pool(x_train, y_train)
            x_test = self.x.iloc[test_ind]
            y_test = self.y.iloc[test_ind]
            pool_test = Pool(x_test, y_test)

            # sample_weight for imbalanced class.
            if self.is_regression():
                sample_weight = None
            else:
                sample_weight = compute_sample_weight(class_weight="balanced",
                                                      y=y_train)

            classifier_obj.fit(pool_train,
                               sample_weight=sample_weight,
                               eval_set=pool_test,
                               verbose=False,
                               early_stopping_rounds=round(
                                   classifier_obj.n_estimators * 0.1) + 2)

            if self.using_earlystopping() and training:
                cv_stoppoint[i] = self.clr_best_iteration(classifier_obj)

            train_score = self.metric(classifier_obj, x_train, y_train)
            test_score = self.metric(classifier_obj, x_test, y_test)

            if self.validate_penalty:
                score[i] = test_score + 0.1 * (test_score - train_score)
            else:
                score[i] = test_score

        # averaging over cv
        #self.thresholds[trial.number] = cv_thresholds.sum() / self.n_cv
        self.stop_points[trial.number] = round(cv_stoppoint.sum() /
                                               (self.n_cv - 1))

        return score.mean()

    def using_earlystopping(self):
        return True

    def _explainer(self, x):
        return shap.TreeExplainer(self.best_model)
