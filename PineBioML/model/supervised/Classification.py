from . import Basic_tuner

from sklearn.model_selection import StratifiedKFold, cross_val_score

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import numpy as np


# linear model
class ElasticNet_tuner(Basic_tuner):
    """
    [sklearn logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression), reminds the choice of the algorithm depends on the penalty chosen and on (multinomial) multiclass support.
    """

    def __init__(self,
                 x,
                 y,
                 kernel="saga",
                 n_try=20,
                 cv=None,
                 target="matthews_corrcoef",
                 kernel_seed=None,
                 optuna_seed=71):
        super().__init__(x,
                         y,
                         n_try=n_try,
                         cv=cv,
                         target=target,
                         kernel_seed=kernel_seed,
                         optuna_seed=optuna_seed)

        # ‘saga’ fast convergence is only guaranteed on features with approximately the same scale. You should do a feature-wise (between sample) normalization before fitting.
        self.kernel = kernel  # sage, lbfgs
        self.penalty = "elasticnet"
        print("Logistic Regression with: ", self.penalty, " penalty, ",
              self.kernel, " solver.")

    def create_model(self, trial):
        C = trial.suggest_float('C', 1e-6, 1e+2, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0, 1)

        lg = LogisticRegression(penalty=self.penalty,
                                C=C,
                                class_weight="balanced",
                                solver=self.kernel,
                                max_iter=100,
                                verbose=0,
                                l1_ratio=l1_ratio)
        return lg

    def evaluate(self, trial):
        classifier_obj = self.create_model(trial)
        score = cross_val_score(
            classifier_obj,
            self.x,
            self.y,
            n_jobs=-1,
            cv=StratifiedKFold(n_splits=5,
                               shuffle=True,
                               random_state=self.seed_recorder[trial.number]),
            scoring=self.metric)
        score = score.mean()
        return score


# RF
class RandomForest_tuner(Basic_tuner):

    def __init__(self,
                 x,
                 y,
                 using_oob=True,
                 n_try=50,
                 cv=None,
                 target="matthews_corrcoef",
                 kernel_seed=None,
                 optuna_seed=71):
        super().__init__(x,
                         y,
                         n_try=n_try,
                         cv=cv,
                         target=target,
                         kernel_seed=kernel_seed,
                         optuna_seed=optuna_seed)

        self.using_oob = using_oob

    def create_model(self, trial):
        parms = {
            "n_estimators":
            trial.suggest_int('n_estimators', 32, 1024, log=True),
            "max_depth":
            trial.suggest_int('max_depth', 2, 32, log=True),
            "min_samples_split":
            trial.suggest_int('min_samples_split', 1, 32, log=True),
            "min_samples_leaf":
            trial.suggest_int('min_samples_leaf', 1, 32, log=True),
            "ccp_alpha":
            trial.suggest_float('ccp_alpha', 1e-4, 1e-1, log=True),
            "max_samples":
            trial.suggest_float('max_samples', 0.5, 0.95, log=True),
            "bootstrap":
            True,
            "oob_score":
            True,
            "n_jobs":
            -1,
            "random_state":
            self.seed_recorder[trial.number],
            "verbose":
            0,
            "class_weight":
            "balanced_subsample"
        }

        rf = RandomForestClassifier(**parms).fit(self.x, self.y)
        return rf

    def evaluate(self, trial):
        classifier_obj = self.create_model(trial)

        if self.using_oob:
            # oob predict
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
                n_jobs=-1,
                cv=StratifiedKFold(
                    n_splits=5,
                    shuffle=True,
                    random_state=self.seed_recorder[trial.number]),
                scoring=self.metric)
            score = score.mean()
        return score


# SVM
class SVC_tuner(Basic_tuner):

    def __init__(self,
                 x,
                 y,
                 kernel="rbf",
                 n_try=20,
                 cv=None,
                 target="matthews_corrcoef",
                 kernel_seed=None,
                 optuna_seed=71):
        super().__init__(x,
                         y,
                         n_try=n_try,
                         cv=cv,
                         target=target,
                         kernel_seed=kernel_seed,
                         optuna_seed=optuna_seed)

        self.n_sample = x.shape[0]
        self.kernel = kernel  # rbf, linear, poly, sigmoid

    def create_model(self, trial):
        # scaling penalty: https://scikit-learn.org/stable/auto_examples/svm/plot_svm_scale_c.html#sphx-glr-auto-examples-svm-plot-svm-scale-c-py
        svc_c = trial.suggest_float('svc_c',
                                    1e-4 * np.sqrt(self.n_sample),
                                    1e+2 * np.sqrt(self.n_sample),
                                    log=True)
        svm = SVC(C=svc_c,
                  kernel=self.kernel,
                  cache_size=1e+3,
                  class_weight="balanced",
                  gamma="auto",
                  probability=True,
                  random_state=self.seed_recorder[trial.number])
        return svm

    def evaluate(self, trial):
        classifier_obj = self.create_model(trial)

        score = cross_val_score(
            classifier_obj,
            self.x,
            self.y,
            n_jobs=-1,
            cv=StratifiedKFold(n_splits=5,
                               shuffle=True,
                               random_state=self.seed_recorder[trial.number]),
            scoring=self.metric)
        score = score.mean()
        return score


# XGboost
# lightGBM
# catboost
# KNN
# KNN-Graph spectrum
