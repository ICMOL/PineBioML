from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
import optuna


# linear model
# RF
class RandomForest_tuner():

    def __init__(self, x, y, target="accuracy"):
        self.x = x
        self.y = y
        #self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=142)
        self.n_try = 50
        self.eval_prob = not target in ["accuracy", "f1"]

        self.metric = metrics.get_scorer(target)

        self.study = optuna.create_study(direction="maximize")

    def evaluate(self, trial):
        parms = {
            "n_estimators":
            trial.suggest_int('n_estimators', 16, 1024, log=True),
            "max_depth":
            trial.suggest_int('max_depth', 2, 32, log=True),
            "min_samples_split":
            trial.suggest_int('min_samples_split', 2, 16, log=True),
            "min_samples_leaf":
            trial.suggest_int('min_samples_leaf', 2, 16, log=True),
            "ccp_alpha":
            trial.suggest_float('ccp_alpha', 1e-4, 1e-1, log=True),
            "max_samples":
            trial.suggest_float('max_samples', 0.5, 1.0, log=True),
            "bootstrap":
            True,
            "oob_score":
            True,
            "n_jobs":
            -1,
            "random_state":
            142,
            "verbose":
            0,
            "class_weight":
            "balanced_subsample"
        }

        classifier_obj = RandomForestClassifier(**parms).fit(self.x, self.y)
        # oob predict
        y_pred = classifier_obj.oob_decision_function_[:, 1]
        # oob score
        if not self.eval_prob:
            y_pred = y_pred > 0.5
        score = self.metric._score_func(self.y, y_pred)

        return score

    def tune(self):
        self.study.optimize(self.evaluate, n_trials=self.n_try)
        parms = {
            "bootstrap": True,
            "oob_score": True,
            "n_jobs": -1,
            "random_state": 142,
            "verbose": 0,
            "class_weight": "balanced_subsample"
        }
        for parameter in self.study.best_params:
            parms[parameter] = self.study.best_params[parameter]

        self.best_model = RandomForestClassifier(**parms)
        return self.best_model


# rbf-SVM
class SVC_tuner():

    def __init__(self, x, y, target="accuracy"):
        self.x = x
        self.y = y
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=142)
        self.n_try = 50
        self.metric = metrics.get_scorer(target)

        self.study = optuna.create_study(direction="maximize")

    def evaluate(self, trial):
        svc_c = trial.suggest_float('svc_c', 1e-6, 1e+2, log=True)
        classifier_obj = SVC(C=svc_c,
                             cache_size=1e+3,
                             class_weight="balanced",
                             gamma="auto",
                             probability=True)

        score = cross_val_score(classifier_obj,
                                self.x,
                                self.y,
                                n_jobs=-1,
                                cv=self.cv,
                                scoring=self.metric)
        score = score.mean()
        return score

    def tune(self):
        self.study.optimize(self.evaluate, n_trials=self.n_try)

        self.best_model = SVC(C=self.study.best_params["svc_c"],
                              probability=True,
                              cache_size=1e+3,
                              class_weight="balanced",
                              gamma="auto")
        return self.best_model


# XGboost
# lightGBM
# catboost
# KNN
# KNN-Graph spectrum
