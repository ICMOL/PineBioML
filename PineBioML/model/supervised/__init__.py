import sklearn.metrics as metrics
import optuna
from abc import ABC, abstractmethod
from sklearn.model_selection import StratifiedKFold
from optuna.samplers import TPESampler
from numpy.random import RandomState

optuna.logging.set_verbosity(optuna.logging.WARNING)


class Basic_tuner(ABC):

    def __init__(self,
                 x,
                 y,
                 n_try=50,
                 cv=None,
                 target="neg_log_loss",
                 kernel_seed=None,
                 optuna_seed=None):
        """
        To reduce the overfitting of optuna on validation set.
        
        Args:
            x: data used to fit
            y: label to fit
            n_try(int): times to try.
            cv(sklearn.model_selection.KFold): A [sklearn cross validation objects](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html)
            target(str): It will pass to sklearn.metrics.get_scorer. Using sklearn.metrics.get_scorer_names() to list available metrics. Notice that 
            kernel_seed(float): random seed for kernel.
            optuna_seed(float): random seed for optuna.
        """
        self.x = x
        self.y = y
        self.cv = cv
        self.n_try = n_try
        self.kernel_seed = kernel_seed
        self.optuna_seed = optuna_seed

        self.discrete_target = ["accuracy", "f1"]
        self.eval_prob = not target in self.discrete_target
        self.metric = metrics.get_scorer(target)

        # Make the sampler behave in a deterministic way.
        sampler = TPESampler(seed=optuna_seed)
        self.study = optuna.create_study(direction="maximize", sampler=sampler)

        self.seed_recorder = RandomState(self.kernel_seed).randint(low=0,
                                                                   high=1024 *
                                                                   16,
                                                                   size=n_try)

    @abstractmethod
    def create_model(self, trial):
        """
        create model based on trial and random_state.

        Args:
            trial(optuna.trial.Trial): optuna trial in this call.
            random_state(float): random seed to use.
            
        Returns :
            sklearn.base.BaseEstimator: A sklearn style model object.
        """
        pass

    @abstractmethod
    def evaluate(self, trial):
        """
        To evaluate the score of this trial. you should call create_model instead create model in this function.
        
        Args:
            trial(optuna.trial.Trial): optuna trial in this call.
            
        Returns :
            float: The score.
        """
        pass

    def tune(self):
        """
        this function should tune the hyperparameters of a given kernel.
        
        Returns :
            sklearn.base.BaseEstimator: A sklearn style model object which has the best hyperparameters.
        """
        print("start tuning. it will take a while.")
        self.study.optimize(self.evaluate, n_trials=self.n_try)

        print("best trial: ", self.study.best_trial.number,
              "  |  optuna seed: ", self.optuna_seed, "  |  model seed: ",
              self.seed_recorder[self.study.best_trial.number])
        print("best parameters: ", self.study.best_params)
        self.best_model = self.create_model(self.study.best_trial)
        return self.best_model