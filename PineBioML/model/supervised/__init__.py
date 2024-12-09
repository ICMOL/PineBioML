import sklearn.metrics as metrics
from sklearn.base import BaseEstimator
import optuna
from abc import ABC, abstractmethod
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder
from optuna.samplers import TPESampler
from numpy.random import RandomState, randint
from pandas import Series, DataFrame, concat

from sklearn.exceptions import ConvergenceWarning

optuna.logging.set_verbosity(optuna.logging.WARNING)
ConvergenceWarning('ignore')


class Basic_tuner(ABC):
    """
    The base class of tuner. A tuner is a wrapper of optuna + models    
    What the tuners do:    
        1. interface of optuna and models with sklearn api style    
        2. randomity management.    
        3. providing a uniform interface to regression and classiciation(binary and multi-class)    

    To conserve the reproducibility and to reduce the hyper parameter overfitting along the process of hyper parameter tuning,    
    we first using valid_seed to randomly initialize a tape of integers and it will sequentially be used in optuna trials.    


    """

    def __init__(self,
                 n_try: int,
                 n_cv: int,
                 target: str,
                 validate_penalty: bool,
                 kernel_seed: int = None,
                 valid_seed: int = None,
                 optuna_seed: int = None):
        """

        Args:
            n_try (int): The number of trials optuna should try.
            n_cv (int): The number of folds to execute cross validation evaluation in iteration of optuna optimization.
            target (str): The target of optuna optimization. Notice that is different from the training loss of model.
            validate_penalty (bool): True to penalty the overfitting by difference between training score and cv score.
            kernel_seed (int, optional): Random seed for model. Defaults to None.
            valid_seed (int, optional): Random seed for cross validation. Defaults to None.
            optuna_seed (int, optional): Random seed for optuna's hyperparameter sampling. Defaults to None.

        ToDo:
            1. transfer the initialization of seed tape from __init__ to fit.    
            2. optuna pruner: See the section Acticating Pruners in https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html
            3. Winner's curse

        """
        self.y_mapping = LabelEncoder()

        self.validate_penalty = validate_penalty

        self.n_cv = n_cv
        self.n_try = n_try
        self.n_sample = 1

        # initialize the random seeds
        if kernel_seed is None:
            self.kernel_seed = randint(16384)
        else:
            self.kernel_seed = kernel_seed

        if valid_seed is None:
            self.valid_seed = randint(16384)
        else:
            self.valid_seed = valid_seed

        if optuna_seed is None:
            self.optuna_seed = randint(16384)
        else:
            self.optuna_seed = optuna_seed

        # The random seed tapes for cross validation along the optuna's optimization trials.
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

    def reference(self) -> dict[str, str]:
        """
        This function will return reference of this method in python dict.    
        If you want to access it in PineBioML api document, then click on the    >Expand source code     

        Returns:
            dict[str, str]: a dict of reference.
        """
        refers = {
            "optuna publication":
            "https://dl.acm.org/doi/10.1145/3292500.3330701",
            "optuna document": "https://optuna.org/",
            "sklearn publication":
            "https://dl.acm.org/doi/10.5555/1953048.2078195"
        }

        return refers

    @abstractmethod
    def name(self) -> str:
        """
        Returns:
            str: Name of this tuner.
        """
        pass

    @abstractmethod
    def is_regression(self) -> bool:
        """
        Returns:
            bool: True if the task of tuner is a regression task.
        """
        pass

    @abstractmethod
    def parms_range(self) -> dict:
        """
        model hyper-parameter search range.

        Returns:
            dict: {parameter_name : (parameter_name, parameter_dtype, lower_bound, upper_bound)}
        """
        pass

    def parms_range_sparser(self, trial, search_setting):
        parameter_name, parameter_dtype, lower_bound, upper_bound = search_setting
        log = lower_bound > 0 and upper_bound / lower_bound > 10

        if parameter_dtype == "float":
            param = trial.suggest_float(parameter_name,
                                        lower_bound,
                                        upper_bound,
                                        log=log)
        elif parameter_dtype == "int":
            param = trial.suggest_int(parameter_name,
                                      lower_bound,
                                      upper_bound,
                                      log=log)
        elif parameter_dtype == "bool":
            param = trial.suggest_categorical(parameter_name,
                                              [lower_bound, upper_bound])
        elif parameter_dtype == "category":
            param = trial.suggest_categorical(parameter_name, lower_bound)
        else:
            raise ValueError(
                "parameter type not support, receaive parameter_dtype {}, parameter_name {}"
                .format(parameter_dtype, parameter_name))
        return param

    @abstractmethod
    def create_model(self, trial, default) -> BaseEstimator:
        """
        Create model based on default setting or optuna trial over search range.

        Args:
            trial (optuna.trial.Trial): optuna trial in this call.
            default (bool): set True to use default hyper parameter
            
        Returns :
            sklearn.base.BaseEstimator: A sklearn style model object.
        """
        pass

    def evaluate(self, trial, default=False) -> float:
        """
        To evaluate the score of this trial. you should call create_model instead of creating model manually in this function.
        
        Args:
            trial (optuna.trial.Trial): optuna trial in this call.
            default (bool): To use default hyper parameter. This argument will be passed to creat_model
        Returns :
            float: The score. Decided by optimization target.
        """
        # create the model using from this trial
        classifier_obj = self.create_model(trial, default)

        # create cross validation
        cv = StratifiedKFold(n_splits=self.n_cv,
                             shuffle=True,
                             random_state=self.valid_seed_tape[trial.number])

        # do cv
        score = []
        for i, (train_ind, test_ind) in enumerate(cv.split(self.x, self.y)):
            # train test split
            x_train = self.x.iloc[train_ind]
            y_train = self.y.iloc[train_ind]
            x_test = self.x.iloc[test_ind]
            y_test = self.y.iloc[test_ind]

            # sample_weight for imbalanced class.
            if self.is_regression():
                sample_weight = None
            else:
                sample_weight = compute_sample_weight(class_weight="balanced",
                                                      y=y_train)

            # fit the model on training fold
            classifier_obj.fit(x_train, y_train, sample_weight=sample_weight)

            # evaluate on testing fold
            test_score = self.metric(classifier_obj, x_test, y_test)
            train_score = self.metric(classifier_obj, x_train, y_train)

            if self.validate_penalty:
                score.append(test_score + 0.1 * (test_score - train_score))
            else:
                score.append(test_score)

        # averaging cv scores
        score = sum(score) / self.n_cv
        #print(score)
        return score

    def tune(self, x, y) -> None:
        """
        this function tunes the hyperparameters of a given kernel.
        
        Args:
            x (pandas.DataFrame or 2D-array): feature to extract information from.
            y (pandas.Series or 1D-array): The property we interested in.

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
        print("    {} start tuning. it will take a while.".format(self.name()))
        # using optuna tuning hyper parameter
        self.study.optimize(self.evaluate,
                            n_trials=self.n_try,
                            show_progress_bar=False)
        self.optuna_model = self.create_model(self.study.best_trial,
                                              default=False)

        # using default hyper parameter
        # the best_trial here is only a placeholder. It's not functional.
        self.default_performance = self.evaluate(self.study.best_trial,
                                                 default=True)
        self.default_model = self.create_model(self.study.best_trial,
                                               default=True)
        #print("    default performance: {:.3f}  |  best performance: {:.3f}".
        #      format(self.default_performance, self.study.best_trial.value))
        if self.default_performance > self.study.best_trial.value:
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
        Set retune to True to retune the model using the given x and y, else using the previously tuned model to fit on given x and y.

        Args:
            x (pandas.DataFrame or 2D-array): feature to extract information from.
            y (pandas.Series or 1D-array): ground true.
            retune (bool): True to retune the model using given x and y, else using the tuned model to fit on given x, y.
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

    def get_scorer(self, scorer_name: str):
        """
        A lazy function to call sklearn scorers.    

        Args:
            scorer_name (str): abbreviation or formal name of sklearn scorers.

        Returns:
            scorer: A callable object that returns score.
        """

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

    def check_task(self) -> bool:
        """
        Uh....  this functions will do    
            1. some check for mis-using of method and types.    
            2. sparsing the task and target.    

        It will be nasty and full of dirty code. Make yourself at home.    
        

        Raises:
            ValueError: Check auc score only be used in binary classification

        Returns:
            bool: _description_
        """
        # Check auc score only be used in binary classification
        is_auc = self.metric_name == "roc_auc"
        is_binary = len(self.y.value_counts()) == 2
        if is_auc and not is_binary:
            # roc_auc only can be used on binary classification. Do not try ovr, ovo. forget them.
            raise ValueError(
                "auc only support binary classification, but more than 2 values are detected in y. Try 'f1_macro'"
            )

        # Sparse the metric
        scorer_kargs = {}
        for arguments in self.metric.__str__()[12:-1].split(", ")[1:]:
            if "(" in arguments:  # for roc auc scorer: response_method=('decision_function', 'predict_proba')
                scorer_kargs["response_method"] = "predict_proba"
            else:
                arg = arguments.split("=")
                if len(arg) == 2:
                    scorer_kargs[arg[0]] = arg[1]

        ### Response method
        self.metric_using_proba = scorer_kargs["response_method"].find(
            "_proba") != -1
        scorer_kargs.pop("response_method")

        ### Greater is better
        if 'greater_is_better' in scorer_kargs:
            self.metric_great_better = not scorer_kargs[
                "greater_is_better"] == "False"
            scorer_kargs.pop('greater_is_better')
        else:
            self.metric_great_better = True

        ### pos_label for f1 socres
        if "pos_label" in scorer_kargs:
            scorer_kargs.pop("pos_label")

        self.scorer_kargs = scorer_kargs

        return True

    def plot(self):
        from plotly import io
        fig = optuna.visualization.plot_optimization_history(
            self.study, target_name=self.metric_name)
        fig.add_hline(y=self.default_performance,
                      line_dash="dot",
                      annotation_text="Default setting",
                      annotation_position="bottom right")
        io.show(fig)

    def detail(self):
        """
        show the experiment settings including:    
            1. models parameters searching range.    

        Returns:
            pandas.DataFrame
        """
        parms_range = DataFrame(self.parms_range()).drop(0).T.reset_index()
        parms_range.columns = [
            "parameter", "dtype", "lower_bound", "upper_bound"
        ]
        parms_range.index = [" "] * parms_range.shape[0]

        name_holder = DataFrame({
            self.name(): {
                "parameter": None,
                "dtype": None,
                "lower_bound": None,
                "upper_bound": None
            }
        }).T
        return concat([name_holder, parms_range], axis=0)
