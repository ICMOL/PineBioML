import sklearn.metrics as metrics
import optuna
from abc import ABC, abstractmethod
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder
from optuna.samplers import TPESampler
from numpy.random import RandomState, randint
from pandas import Series, DataFrame

from sklearn.exceptions import ConvergenceWarning

optuna.logging.set_verbosity(optuna.logging.WARNING)
ConvergenceWarning('ignore')


class Basic_tuner(ABC):
    """
    The base class of tuner.    
    To conserve the reproducibility and to reduce the hyper parameter overfitting along the process of hyper parameter tuning, 
    we first using valid_seed to randomly initialize a tape of integers and it will sequentially be used in optuna trials.    


    """

    def __init__(self,
                 n_try,
                 n_cv,
                 target,
                 validate_penalty,
                 kernel_seed=None,
                 valid_seed=None,
                 optuna_seed=None):
        """
        
        Args:
            n_try (int): Times to try.    
            target (str): The target of hyperparameter tuning. It will pass to sklearn.metrics.get_scorer . Using sklearn.metrics.get_scorer_names() to list available metrics.    
            kernel_seed (int): random seed for model kernel.    
            valid_seed (int): random seed for cross validation    
            optuna_seed (int): random seed for optuna.    

        ToDo: None seed for tuner from __init__ to fit    
        """
        self.y_mapping = LabelEncoder()

        self.validate_penalty = validate_penalty

        self.n_cv = n_cv
        self.n_try = n_try

        if kernel_seed:
            self.kernel_seed = kernel_seed
        else:
            self.kernel_seed = randint(16384)

        if valid_seed:
            self.valid_seed = valid_seed
        else:
            self.valid_seed = randint(16384)

        if optuna_seed:
            self.optuna_seed = optuna_seed
        else:
            self.optuna_seed = randint(16384)

        # The tape for cross validation random seed along optuna trial.
        self.valid_seed_tape = RandomState(self.valid_seed).randint(low=0,
                                                                    high=16384,
                                                                    size=n_try)
        self.kernel_seed_tape = RandomState(self.kernel_seed).randint(
            low=0, high=16384, size=n_try)

        # Get the scorer
        self.metric = self.get_scorer(target)

        self.optuna_model = None
        self.default_model = None
        self.best_model = None

    @abstractmethod
    def is_regression(self):
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

    def evaluate(self, trial, default=False):
        """
        To evaluate the score of this trial. you should call create_model instead of creating model manually in this function.
        
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
            x_test = self.x.iloc[test_ind]
            y_test = self.y.iloc[test_ind]

            if self.is_regression():
                sample_weight = None
            else:
                sample_weight = compute_sample_weight(class_weight="balanced",
                                                      y=y_train)

            classifier_obj.fit(x_train, y_train, sample_weight=sample_weight)

            test_score = self.metric(classifier_obj, x_test, y_test)
            train_score = self.metric(classifier_obj, x_train, y_train)
            if self.validate_penalty:
                score.append(test_score + 0.1 * (test_score - train_score))
            else:
                score.append(test_score)

        score = sum(score) / self.n_cv
        #print(score)
        return score

    def tune(self, x, y):
        """
        this function should tune the hyperparameters of a given kernel.
        
        Args:
            x (pandas.DataFrame or 2D-array): feature to extract information from.
            y (pandas.Series or 1D-array): ground true.

        Returns :
            sklearn.base.BaseEstimator: A sklearn style model object which has the best hyperparameters.
        """
        # input data
        self.x = x
        self.y = y

        # the total number of samples
        self.n_sample = x.shape[0]

        # check task
        self.check_task()

        # Make the sampler behave in a deterministic way.
        sampler = TPESampler(seed=self.optuna_seed)
        self.study = optuna.create_study(direction="maximize", sampler=sampler)

        print(
            "optuna seed {self.optuna_seed}  |  validation seed {self.valid_seed}  |  model seed {self.kernel_seed}"
            .format(self=self))
        print("    start tuning. it will take a while.")
        # using optuna tuning hyper parameter
        self.study.optimize(self.evaluate,
                            n_trials=self.n_try,
                            show_progress_bar=False)
        self.optuna_model = self.create_model(self.study.best_trial,
                                              default=False)

        # using default hyper parameter
        # the best_trial here is only a placeholder. It's not functional.
        default_performance = self.evaluate(self.study.best_trial,
                                            default=True)
        self.default_model = self.create_model(self.study.best_trial,
                                               default=True)
        #print("    default performance: {:.3f}  |  best performance: {:.3f}".
        #      format(default_performance, self.study.best_trial.value))
        if default_performance > self.study.best_trial.value:
            # default better
            print("    default is better.")
            self.best_model = self.default_model
        else:
            # optuna better
            print("    optuna is better, best trial: ",
                  self.study.best_trial.number)
            self.best_model = self.optuna_model

    def fit(self, x, y, retune=True):
        """
        The sklearn.base.BaseEstimator fit api.

        Args:
            x (pandas.DataFrame or 2D-array): feature to extract information from.
            y (pandas.Series or 1D-array): ground true.
        """
        self.label_name = y.name

        # label encoding
        if not self.is_regression():
            y = Series(self.y_mapping.fit_transform(y),
                       index=y.index,
                       name=y.name)

        # tune the model.
        if retune:
            self.tune(x, y)

        # fit the model.
        if self.is_regression():
            sample_weight = None
        else:
            sample_weight = compute_sample_weight(class_weight="balanced", y=y)
        self.best_model.fit(x, y, sample_weight=sample_weight)

        return self

    def predict(self, x):
        """
        The sklearn.base.BaseEstimator predict api.

        Args:
            x (pandas.DataFrame or 2D-array): feature to extract information from.

        Returns:
            1D-array: prediction
        """
        # using the model.
        y_pred = self.best_model.predict(x)
        # label decoding
        if not self.is_regression():
            y_pred = self.y_mapping.inverse_transform(y_pred)
        y_pred = Series(y_pred, index=x.index, name=self.label_name)

        return y_pred

    def get_scorer(self, scorer_name):
        # easy query
        ### polymorphism
        scorer_name = scorer_name.lower().replace("-", "_").replace(" ", "_")

        ### common abbreviation
        nicknames = {
            "acc": "accuracy",
            "auc": "roc_auc",
            "f1_score": "f1",
            "macro_f1": "f1_macro",
            "mcc": "matthews_corrcoef",
            "log_loss": "neg_log_loss",
            "cross_entropy": "neg_log_loss",
            "ce": "neg_log_loss",
            "bce": "neg_log_loss",
            "mse": "neg_mean_squared_error",
            "mean_squared_error": "neg_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "mean_absolute_error": "neg_mean_absolute_error",
            "mape": "neg_mean_absolute_percentage_error",
            "mean_absolute_percentage_error":
            "neg_mean_absolute_percentage_error",
            "rmse": "neg_root_mean_squared_error",
            "root_mean_squared_error": "neg_root_mean_squared_error",
            "qwk": "quadratic_weighted_kappa",
            "kappa": "cohen_kappa"
        }
        if scorer_name in nicknames:
            scorer_name = nicknames[scorer_name]

        self.metric_name = scorer_name

        # get the scorer
        ### kappa
        if scorer_name == "quadratic_weighted_kappa":
            return metrics.make_scorer(metrics.cohen_kappa_score,
                                       weights="quadratic",
                                       response_method="predict",
                                       greater_is_better=True)
        elif scorer_name == "cohen_kappa":
            return metrics.make_scorer(metrics.cohen_kappa_score,
                                       weights=None,
                                       response_method="predict",
                                       greater_is_better=True)
        ### others
        return metrics.get_scorer(scorer_name)

    def check_task(self):
        # check auc and binary classification
        is_auc = self.metric_name == "roc_auc"
        is_binary = len(self.y.value_counts()) == 2
        if is_auc and not is_binary:
            # roc_auc only can be used on binary classification. Do not try ovr, ovo. forget them.
            raise ValueError(
                "auc only support binary classification, but more than 2 values are detected in y."
            )

        # sparse the metric
        scorer_kargs = {}
        for arg in [
                i.split("=")
                for i in self.metric.__str__()[12:-1].split(", ")[1:]
        ]:
            scorer_kargs[arg[0]] = arg[1]

        # response method
        self.metric_using_proba = scorer_kargs["response_method"].find(
            "_proba") != -1
        scorer_kargs.pop("response_method")

        # greater is better
        if 'greater_is_better' in scorer_kargs:
            self.metric_great_better = not scorer_kargs[
                "greater_is_better"] == "False"
            scorer_kargs.pop('greater_is_better')
        else:
            self.metric_great_better = True

        # pos_label for f1 socres
        if "pos_label" in scorer_kargs:
            scorer_kargs.pop("pos_label")

        self.scorer_kargs = scorer_kargs

        return True
