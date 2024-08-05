import sklearn.metrics as metrics
import optuna
from abc import ABC, abstractmethod
from sklearn.model_selection import StratifiedKFold
from optuna.samplers import TPESampler
from numpy.random import RandomState, randint
from pandas import Series, DataFrame

optuna.logging.set_verbosity(optuna.logging.WARNING)


class Basic_tuner(ABC):
    """
    The base class of tuner.    
    To conserve the reproducibility and to reduce the hyper parameter overfitting along the process of hyper parameter tuning. 
    We first using valid_seed to randomly initialize a tape of integers and it will sequentially be used in optuna trials.    

    """

    def __init__(self, n_try, target, kernel_seed, valid_seed, optuna_seed):
        """
        
        Args:
            n_try (int): Times to try.    
            target (str): The target of hyperparameter tuning. It will pass to sklearn.metrics.get_scorer . Using sklearn.metrics.get_scorer_names() to list available metrics.    
            kernel_seed (int): random seed for model kernel. 
            valid_seed (int): random seed for cross validation
            optuna_seed (int): random seed for optuna.    
        """
        self.n_cv = 5
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

        # If the tuning target is not in discrete_target, the kernel will not be required to return in prob which is costy fro some method like svm.
        self.discrete_target = ["accuracy", "f1", "matthews_corrcoef"]
        self.eval_prob = not target in self.discrete_target
        # Get the scorer
        self.metric = metrics.get_scorer(target)

        # Make the sampler behave in a deterministic way.
        sampler = TPESampler(seed=optuna_seed)
        self.study = optuna.create_study(direction="maximize", sampler=sampler)

        self.optuna_model = None
        self.default_model = None
        self.best_model = None

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

            classifier_obj.fit(x_train, y_train)

            test_score = self.metric(classifier_obj, x_test, y_test)
            train_score = self.metric(classifier_obj, x_train, y_train)
            score.append(test_score + 0.2 * (test_score - train_score))

            #score.append(test_score)
        score = sum(score) / self.n_cv
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
        self.x = x
        self.y = y
        self.n_sample = x.shape[0]

        print(
            "optuna seed {self.optuna_seed}  |  validation seed {self.valid_seed}  |  model seed {self.kernel_seed}"
            .format(self=self))
        print("    start tuning. it will take a while.")
        # using optuna tuning hyper parameter
        self.study.optimize(self.evaluate, n_trials=self.n_try)
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

        print(self.best_model, "\n")
        return self.best_model

    def fit(self, x, y):
        """
        The sklearn.base.BaseEstimator fit api.

        Args:
            x (pandas.DataFrame or 2D-array): feature to extract information from.
            y (pandas.Series or 1D-array): ground true.
        """
        self.label_name = y.name

        # tune the model.
        self.tune(x, y)

        # fit the model.
        self.best_model.fit(x, y)

        return self.best_model

    def predict(self, x):
        """
        The sklearn.base.BaseEstimator predict api.

        Args:
            x (pandas.DataFrame or 2D-array): feature to extract information from.

        Returns:
            1D-array: prediction
        """
        # using the model.
        y_pred = Series(self.best_model.predict(x),
                        index=x.index,
                        name=self.label_name)

        return y_pred

    def predict_proba(self, x):
        """
        The sklearn.base.BaseEstimator predict_prob api.

        Args:
            x (pandas.DataFrame or 2D-array): feature to extract information from.

        Returns:
            1D-array: prediction in prob
        """
        return self.best_model.predict_proba(x)
