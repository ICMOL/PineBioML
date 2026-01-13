import warnings
import sklearn.metrics as metrics
from sklearn.base import BaseEstimator
import optuna
from abc import ABC, abstractmethod
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder
from optuna.samplers import TPESampler
from numpy.random import RandomState, randint
from numpy import zeros
from pandas import Series, DataFrame, concat
from shap import Explainer

from sklearn.exceptions import ConvergenceWarning

optuna.logging.set_verbosity(optuna.logging.ERROR)
ConvergenceWarning('ignore')
"""
class Binary_threshold_wrapper(ClassifierMixin):

    def __init__(self, kernel, threshold):
        self.threshold = threshold
        self.kernel = kernel
        self.classes_ = kernel.classes_

    def fit(self, x, y):
        pass

    def predict(self, x):
        decision = self.kernel.predict_proba(x)[:, 1] > self.threshold
        return decision.astype(int)

    def predict_proba(self, x):
        return self.kernel.predict_proba(x)
"""


class Basic_tuner(ABC, BaseEstimator):
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
                 TT_coef: float,
                 kernel_seed: int = None,
                 valid_seed: int = None,
                 optuna_seed: int = None):
        """

        Args:
            n_try (int): The number of trials optuna should try.
            n_cv (int): The number of folds to execute cross validation evaluation in iteration of optuna optimization.
            target (str): The target of optuna optimization. Notice that is different from the training loss of model.
            validate_penalty (bool): Deprecated.
            TT_coef (float): The power of penalty to overfitting by Tibshirani & Tibshirani method. Ranges in [0, 1]. Add the difference between training score and cv score to the optimization target.
            kernel_seed (int, optional): Random seed for model. Defaults to None.
            valid_seed (int, optional): Random seed for cross validation. Defaults to None.
            optuna_seed (int, optional): Random seed for optuna's hyperparameter sampling. Defaults to None.

        ToDo:
            1. transfer the initialization of seed tape from __init__ to fit.    
            2. optuna pruner: See the section Acticating Pruners in https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html
            3. Winner's curse

        """
        self.y_mapping = LabelEncoder()

        if validate_penalty == True:
            warnings.warn(
                "validate_penalty will be remove in future. Use argument TT_coef.",
                DeprecationWarning,
                stacklevel=2)
            self.TT_coef = TT_coef
        else:
            self.TT_coef = 0

        self.fitted_ = None
        self.n_cv = n_cv
        self.optuna_early_stop_counter = n_cv // 10 + 2
        self.n_try = n_try
        self.n_sample = 1
        self.n_opt_jobs = 1
        self.default = False
        self.training = True

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

        # the model optuna tuned
        self.optuna_model = None
        # the model with default params
        self.default_model = None
        # the better one of self.default_model or self.optuna_model
        self.best_model = None
        # the thresholds for binary classification along optuna tuning
        #self.thresholds = zeros(self.n_try) + 0.5
        # the threshold coresponding to the best optuna trial
        #self.best_threshold = 0.5
        # the stopping point of early stopping for Boosting methods.
        self.stop_points = zeros(self.n_try, dtype=int)

        self.explainer = None

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
        To be determined.

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

    def using_earlystopping(self) -> bool:
        """
        Returns:
            bool: True if applying earlystopping in optimizing training.
        """
        return False

    def accepting_categorical_features(self) -> bool:
        """
        Returns:
            bool: Ture if kernel receives categorical (discreate) features.
        """
        return False

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
        param = None

        if parameter_dtype in ["float", "int"]:
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
        else:
            if parameter_dtype == "bool":
                param = trial.suggest_categorical(parameter_name,
                                                  [lower_bound, upper_bound])
            elif parameter_dtype == "category":
                param = trial.suggest_categorical(parameter_name, lower_bound)
        if param is None:
            raise ValueError(
                "parameter type not support, receaive parameter_dtype {}, parameter_name {}. Only one of [\"float, int, bool, category] is supported."
                .format(parameter_dtype, parameter_name))
        return param

    @abstractmethod
    def create_model(self, trial, default, training) -> BaseEstimator:
        """
        Create model based on default setting or optuna trial over search range.

        Args:
            trial (optuna.trial.Trial): optuna trial in this call.
            default (bool): set True to use default hyper parameter
            training (bool): set True to when training.
            
        Returns :
            sklearn.base.BaseEstimator: A sklearn style model object.
        """
        pass

    def optimize_fit(self, clr, train_data, sample_weight, valid_data):
        """
        optimize_fit the polymorphism middle layer between model fitting and optuna optimize evaluate.    
        Specifically, optimize_fit is used for XGBoost/lightGBM/Catboost early stopping which requires validation data in .fit .
        However sklearn estimators such as randomforest does not provide such api.    

        Args:
            clr (classifier): classifier to fit.
            train_data (tuple): (train_x, train_y) .
            sample_weight (list-like): training sample weight.
            valid_data (tuple): (valid_x, valid_y) .

        Returns:
            sklearn.base.BaseEstimator: A fitted sklearn style model object.
        """
        train_x, train_y = train_data
        valid_x, valid_y = valid_data

        return clr.fit(train_x, train_y, sample_weight=sample_weight)

    def clr_best_iteration(self, classifier) -> int:
        """
        the polymorphism to call classifier's early stopping rounds/iteration/point.    
        only need to be overide when subclass is implemented with early stopping.

        Args:
            classifier: the classifier object.

        Returns:
            int: the stopping point/rounds/iteration.
        """
        pass

    def evaluate(self, trial, default=None, training=None) -> float:
        """
        To evaluate the score of this trial. you should call create_model instead of creating model manually in this function.
        
        Args:
            trial (optuna.trial.Trial): optuna trial in this call.
            default (bool): To use default hyper parameter. This argument will be passed to creat_model .    
            training (bool): whether it is under optuna optimization or not. False then evalutate will not tune a threshold or applying early stop. 
        Returns :
            float: The score. Decided by optimization target.
        """
        if default is None:
            default = self.default
        if training is None:
            training = self.training

        # create the model using from this trial
        classifier_obj = self.create_model(trial, default, training=True)

        # create cross validation
        if self.is_regression():
            cv = KFold(n_splits=self.n_cv,
                       shuffle=True,
                       random_state=self.valid_seed_tape[trial.number])
        else:
            cv = StratifiedKFold(
                n_splits=self.n_cv,
                shuffle=True,
                random_state=self.valid_seed_tape[trial.number])

        # do cv
        score = zeros(self.n_cv)
        # thresholds not None only if  self.is_binary and not self.is_regression
        cv_thresholds = zeros(self.n_cv) + 0.5
        cv_stoppoint = zeros(self.n_cv)
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
            fitted_clr = self.optimize_fit(clr=classifier_obj,
                                           train_data=(x_train, y_train),
                                           sample_weight=sample_weight,
                                           valid_data=(x_test, y_test))

            # record the stopping point of early stopping.
            if self.using_earlystopping() and training:
                cv_stoppoint[i] = self.clr_best_iteration(fitted_clr)

            # tune a threshold via roc for Binary classification
            """
            if self.is_binary and not self.is_regression():
                if training:
                    fpr, tpr, thr = metrics.roc_curve(
                        y_test,
                        fitted_clr.predict_proba(x_test)[:, 1])
                    ### TODO: flexible threshold picker for various metrics.
                    cv_thresholds[i] = thr[abs(tpr - fpr).argmax()]
                else:
                    cv_thresholds[i] = self.thresholds[trial.number]
                # wrap the classifier before calculate scores
                if not default:
                    fitted_clr = Binary_threshold_wrapper(
                        fitted_clr, cv_thresholds[i])
                else:
                    fitted_clr = Binary_threshold_wrapper(fitted_clr, 0.5)
            """

            # evaluate on testing fold
            test_score = self.metric(fitted_clr, x_test, y_test)
            train_score = self.metric(fitted_clr, x_train, y_train)

            score[i] = test_score + self.TT_coef * (test_score - train_score)

        # averaging over cv
        if training:
            #self.thresholds[trial.number] = cv_thresholds.sum() / self.n_cv
            self.stop_points[trial.number] = round(cv_stoppoint.sum() /
                                                   (self.n_cv - 1))
        return score.mean()

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
        sampler = TPESampler(seed=self.optuna_seed,
                             multivariate=True,
                             n_startup_trials=self.n_try // 3)
        self.study = optuna.create_study(direction="maximize", sampler=sampler)

        print(
            "optuna seed {self.optuna_seed}  |  validation seed {self.valid_seed}  |  model seed {self.kernel_seed}"
            .format(self=self))
        print("    {} start tuning. it will take a while.".format(self.name()))
        # using optuna tuning hyper parameter
        self.training = True
        self.default = False
        self.study.optimize(self.evaluate,
                            n_trials=self.n_try,
                            show_progress_bar=False,
                            callbacks=[self.early_stop_callback],
                            n_jobs=self.n_opt_jobs)
        self.optuna_model = self.create_model(self.study.best_trial,
                                              default=False,
                                              training=False)

        # using default hyper parameter
        # the best_trial here is only a placeholder. It's not functional.
        self.default_performance = self.evaluate(self.study.best_trial,
                                                 default=True,
                                                 training=False)
        self.default_model = self.create_model(self.study.best_trial,
                                               default=True,
                                               training=False)
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
            """
            # threshold tuner for binary classification
            if self.is_binary and not self.is_regression():
                self.best_threshold = self.thresholds[
                    self.study.best_trial.number]
            """

        # release the datas
        self.x = None
        self.y = None

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
        self.fitted_ = True

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
        """
        if self.is_binary and not self.is_regression():
            y_pred = (self.best_model.predict_proba(x)[:, 1]
                      > self.best_threshold).astype(int)
        else:
        """
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
        self.is_binary = len(self.y.value_counts()) == 2
        if is_auc and not self.is_binary:
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
        self.metric_limit = 1 if self.metric_great_better else 0

        ### pos_label for f1 socres
        if "pos_label" in scorer_kargs:
            scorer_kargs.pop("pos_label")

        self.scorer_kargs = scorer_kargs

        return True

    def plot(self):
        """
        Plot the learning process.
        """
        from plotly import io
        fig = optuna.visualization.plot_optimization_history(
            self.study, target_name=self.metric_name)
        fig.add_hline(y=self.default_performance,
                      line_dash="dot",
                      annotation_text="Default setting",
                      annotation_position="bottom right")
        io.show(fig)

    def _explainer(self, x: DataFrame) -> Explainer:
        print("Not implement")
        return None

    def shap_explain(self, x: DataFrame) -> DataFrame:
        """

        Args:
            x (DataFrame): the data to be explained which has shape (Samples, Features).
        
        Returns:
            explainer (shap.Explaination): Indexing by (Features, Output, Samples)
        """
        self.explainer = self._explainer(x)
        return self.explainer(x)

    def early_stop_callback(self, study, trial):
        if trial.value >= self.metric_limit:
            self.optuna_early_stop_counter -= 1
            if self.optuna_early_stop_counter <= 0:
                print("optuna evaluate value {} reachs {}'s maximun value {}".
                      format(study.best_value, self.metric_name,
                             self.metric_limit))
                study.stop()

    def detail(self):
        """
        show the experiment settings including:    
            1. models parameters searching range.    

        Returns:
            pandas.DataFrame
        """
        parms_range = DataFrame(self.parms_range()).drop(0)
        if self.best_model is not None:
            best_params = self.best_model.get_params()

            parms_range.loc[4] = Series(
                [best_params[param] for param in parms_range.columns],
                index=parms_range.columns)

        parms_range = parms_range.T.reset_index()
        parms_range.columns = [
            "parameter", "dtype", "lower_bound", "upper_bound", "result"
        ]

        parms_range.index = [" "] * parms_range.shape[0]

        name_holder = DataFrame({
            self.name(): {
                "parameter": None,
                "dtype": None,
                "lower_bound": None,
                "upper_bound": None,
                "result": None
            }
        }).T
        return concat([name_holder, parms_range], axis=0)
