from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from pandas import DataFrame, Series, concat
from sklearn.base import is_classifier, is_regressor
import time

#TODO: grouped cv


class data_source():
    """
    The data_source is a placeholder for various data source. it will override the data flow of Pine.
    Notice that the label y and the order should the same for each datasource.
    """

    def __init__(
        self,
        train_x: DataFrame,
        test_x: DataFrame = None,
    ):
        self.train_x = train_x
        self.test_x = test_x

    def fit(self, x: DataFrame, y: Series):

        return self

    def transform(self, x: DataFrame):

        return self.test_x

    def fit_transform(self, x: DataFrame, y: Series):

        return self.train_x


class sklearn_esitimator_wrapper():
    """
    A basic wrapper for sklearn_esitimator. It transfer the data pipeline of sklearn from numpy.array to pandas.DataFrame.    
    If you want to pass any model with api in sklearn style into Pine, you should wrap it in wrapper.
    """

    def __init__(self, kernel: object):
        """

        Args:
            kernel (object): a sklearn esitimator. for example: sklearn.ensemble.RandomForestClassifier or sklearn.ensemble.RandomForestRegressor
        """

        self.kernel = kernel

    def fit(self, x: DataFrame, y: Series, retune=True) -> object:
        """
        sklearn esitimator api: fit

        Args:
            x (DataFrame): feature
            y (Series): label
            retune (bool, optional): To retune the model or not. For sklearn_esitimator_wrapper, it is just a placeholder without acutual facility. Defaults to True.

        Returns:
            object: A sklearn_esitimator within pandas data flow.
        """
        self.label_name = y.name
        self.kernel.fit(x, y)
        return self

    def predict(self, x: DataFrame) -> Series:
        """
        sklearn esitimator api: predict

        Args:
            x (DataFrame): feature

        Returns:
            Series: kernel prediction
        """

        return Series(self.kernel.predict(x),
                      index=x.index,
                      name=self.label_name)

    def predict_proba(self, x: DataFrame) -> DataFrame:
        """
        sklearn esitimator api: predict_proba for classification

        Args:
            x (DataFrame): feature

        Raises:
            NotImplementedError: Regression has no attribute 'predict_proba'

        Returns:
            DataFrame: predicted probability with shape (n_sample, n_class)
        """
        if "predict_proba" in dir(self.kernel):
            return DataFrame(self.kernel.predict_proba(x),
                             index=x.index,
                             columns=self.kernel.classes_)
        else:
            raise NotImplementedError(
                "{} do not have attribute 'predict_proba'.".format(
                    self.kernel.__str__()))

    def is_regression(self) -> bool:
        return is_regressor(self.kernel)

    def detail(self):
        return None


class classification_scorer():
    """
    A utility to calculate classification scores.    
    The result will contain mcc(matthews corrcoef), acc(accuracy) and support(the number of samples), furthermore:    
        if target_label was given(not None), then sensitivity, specificity and coresponding roc-auc score will be added to result.    
        if multi_class_extra is True, then one vs rest macro_auc, cross_entropy and cohen_kappa will be added to result.    
    """

    def __init__(self,
                 target_label: str = None,
                 prefix: str = "",
                 multi_class_extra: bool = False):
        """

        Args:
            target_label (str, optional): the name of target_label. For example, the label in a binary classification task might be {'pos', 'neg'}. Then you can assign 'neg' to target_label, and the result will contain sensitivity, specificity and roc-auc score of label 'neg'. Defaults to None.
            prefix (str, optional): prefix before score names. For example suppose prefix="Train_", then all the scores in result will be like "Train_mcc". Defaults to "".
            multi_class_extra (bool, optional): _description_. Defaults to False.
        """

        self.prefix = prefix
        self.target_label = target_label
        self.multi_class_extra = multi_class_extra

    def score(self,
              y_true: Series,
              y_pred_prob: DataFrame,
              y_pred: Series = None) -> dict[str, float]:
        """
        Scoring y_true and y_pred_prob.

        Args:
            y_true (Series): The ground True.
            y_pred_prob (DataFrame): The prediction from an estimator. Shape should be (n_sample, n_classes)
            y_pred (Series, optional): The prediction made by model. For Binary classification models, the prediction may differ from prob.argmax because of threshold tuning. Defaults to None.

        Returns:
            dict[str, float]: The result stored in a dict, be like {'score_name': score}.
        """
        if y_pred is None:
            y_pred = y_pred_prob.idxmax(axis=1)

        result = {}
        if not self.target_label is None:
            (_, result["sensitivity"], result["f1"],
             _) = metrics.precision_recall_fscore_support(
                 y_true=(y_true == self.target_label),
                 y_pred=(y_pred == self.target_label),
                 average="binary",
                 pos_label=True)

            (_, result["specificity"], _,
             _) = metrics.precision_recall_fscore_support(
                 y_true=(y_true == self.target_label),
                 y_pred=(y_pred == self.target_label),
                 average="binary",
                 pos_label=False)

            # binary
            result["auc"] = metrics.roc_auc_score(
                y_true == self.target_label, y_pred_prob[self.target_label])

        if self.multi_class_extra:
            result["macro_auc"] = metrics.roc_auc_score(
                y_true,
                y_pred_prob,
                multi_class="ovr",
                labels=y_pred_prob.columns)
            result["macro_f1"] = metrics.f1_score(y_true,
                                                  y_pred,
                                                  average="macro")
            result["cross_entropy"] = metrics.log_loss(y_true, y_pred_prob)
            result["cohen_kappa"] = metrics.cohen_kappa_score(y_true, y_pred)

        result["mcc"] = metrics.matthews_corrcoef(y_true, y_pred)
        result["accuracy"] = metrics.accuracy_score(y_true, y_pred)
        result["support"] = len(y_true)

        prefix_result = {}
        for score in result:
            prefix_result[self.prefix + score] = result[score]

        return prefix_result


class regression_scorer():
    """
    A utility to calculate regression scores. rmse(rooted mean squared error), r2(R squared) and support(the number of samples) are included.    
    if y_true and y_pred are all positive, then mape(mean absolute percentage error) will be added.    
    """

    def __init__(self, prefix: str = "", target_label: str = None):
        """

        Args:
            prefix (str, optional): prefix before score names. For example suppose prefix="Train_", then all the scores in result will be like "Train_mse". Defaults to "".
            target_label (str, optional): A placehold without any facility. Defaults to None.
        """

        self.prefix = prefix

    def score(self,
              y_true: Series,
              y_pred: Series,
              place_holder=None) -> dict[str, float]:
        """
        calculate the scores

        Args:
            y_true (Series): Ground true.
            y_pred (Series): predicted values.
            place_holder (None): A placeholder corresponding to classification_scorer's pred argument.

        Returns:
            dict[str, float]: The result stored in a dict, be like {'score_name': score}.
        """

        result = {}
        result["rmse"] = metrics.root_mean_squared_error(y_true, y_pred)
        result["r2"] = metrics.r2_score(y_true, y_pred)
        if (y_true > 0).all() and (y_pred > 0).all():
            result["mape"] = metrics.mean_absolute_percentage_error(
                y_true, y_pred)
        result["support"] = len(y_true)

        prefix_result = {}
        for score in result:
            prefix_result[self.prefix + score] = result[score]

        return prefix_result


# ToDo: the Integration of .predict and .transform
class Pine():
    """
    Deep first traversal the given experiment setting.    
    the last step of experiment sould be model.    
    Please refer to example_Pine.ipynb for usage.    


    note: experiment step and experiment stage is the same thing.    
    """

    def __init__(self,
                 experiment: list[tuple[str, dict[str, object]]],
                 target_label: str = None,
                 cv_result: bool = False,
                 evaluate_ncv: int = 5):
        """
        Args:
            experiment (list[tuple[str, dict[str, object]]]): list of experiment steps. step should be in the form: ('step_name', {'method_name': method}). it could be several method in one step and they will fork in deep first traversal. Each method should be either sklearn estimator or transformer.
            target_label (str, optional): the name of target_label. For example, the label in a binary classification task might be {'pos', 'neg'}. Then you can assign 'neg' to target_label, and the result will contain sensitivity, specificity and roc-auc score of label 'neg'. Defaults to None.
            cv_result (bool, optional): Rcording the scores and prediction of cross validation. Defaults to False.
            evaluate_cv (int, optional): The number of folds to evaluate cv_result after pipeline tuned. Defaults to 5.
        """

        self.experiment = experiment
        self.total_stage = len(experiment)
        self.target_label = target_label
        self.cv_result = cv_result
        self.evaluate_ncv = evaluate_ncv

        self.result = []

        self.train_pred = []
        self.cv_pred = []
        self.test_pred = []

    def do_stage(self, train_x: DataFrame, train_y: Series, test_x: DataFrame,
                 test_y: Series, stage: int, record_path: dict,
                 record_time: dict) -> None:
        """
        the recursive function to traversal the experiment.    
        the socres and path will be stored in self.result amd self.____pred, so there is no return in recursive function.     

        Args:
            train_x (pd.DataFrame): training x
            train_y (pd.Series): training y
            test_x (pd.DataFrame): training x
            test_y (pd.Series): training y
            stage (int): the order of current stage in the experiment setting
            record_path (dict): record_path the traversal path in a dict of str
            record_time (dict): record_time the traversal time in a dict of str
        """

        # unzip the stage, stage = (stage_name, {operator_name: operator})
        stage_name, operators = self.experiment[stage]

        # fork to next stage according to the diffirent operator (opt)
        for opt_name in operators:
            record_path[stage_name] = opt_name

            opt = operators[opt_name]

            # if not the last stage
            if stage < self.total_stage - 1:
                time_start = time.time()

                # transform by operators
                processed_train_x = opt.fit_transform(train_x, train_y)
                if test_x is not None:
                    processed_test_x = opt.transform(test_x)
                else:
                    processed_test_x = test_x

                time_end = time.time()
                record_time[stage_name + "_time"] = time_end - time_start
                # reccursivly call
                self.do_stage(processed_train_x, train_y, processed_test_x,
                              test_y, stage + 1, record_path, record_time)

            # the last layer, it should be models
            else:

                model = opt
                if "predict_proba" in dir(model):
                    # is not regression
                    f = model.predict_proba
                    scorer = classification_scorer
                else:
                    # is regression
                    f = model.predict
                    scorer = regression_scorer
                time_start = time.time()
                # tune/fit the model on training data
                model.fit(train_x, train_y)
                time_end = time.time()
                record_time[stage_name + "_fit_time"] = time_end - time_start

                # compute the training score
                time_start = time.time()
                train_pred = f(train_x)
                time_end = time.time()
                record_time[stage_name +
                            "_predict_time"] = time_end - time_start

                # compute the prediction for those who has a tuned threshold in binary classification task.
                train_prediction = model.predict(train_x)

                self.train_pred.append(train_pred)
                train_scores = scorer(prefix="train_",
                                      target_label=self.target_label).score(
                                          train_y, train_pred,
                                          train_prediction)

                if test_x is not None:
                    # if there is testing data, compute the testing score.
                    test_pred = f(test_x)
                    test_prediction = model.predict(test_x)
                    self.test_pred.append(test_pred)
                    test_scores = scorer(prefix="test_",
                                         target_label=self.target_label).score(
                                             test_y, test_pred,
                                             test_prediction)
                else:
                    test_scores = {}

                if self.cv_result:
                    # compute the cross validation score on training set
                    fold_scores = []
                    cv_pred = []

                    if model.is_regression():
                        cross_validation = KFold(n_splits=self.evaluate_ncv,
                                                 shuffle=True,
                                                 random_state=133)
                    else:
                        cross_validation = StratifiedKFold(
                            n_splits=self.evaluate_ncv,
                            shuffle=True,
                            random_state=133)

                    for (train_idx, valid_idx) in cross_validation.split(
                            train_x, train_y):

                        # fit on training fold
                        model.fit(train_x.iloc[train_idx],
                                  train_y.iloc[train_idx],
                                  retune=False)

                        # score on testing fold
                        fold_pred = f(train_x.iloc[valid_idx])
                        #fold_prediction = model.predict(train_x.iloc[valid_idx])

                        cv_pred.append(fold_pred)
                        fold_scores.append(
                            scorer(prefix="cv_",
                                   target_label=self.target_label).score(
                                       train_y.iloc[valid_idx], fold_pred))
                    # average the fold scores
                    self.cv_pred.append(concat(cv_pred, axis=0))
                    valid_scores = DataFrame(fold_scores).mean().to_dict()
                    # TODO accurate statistic estimate of std.
                    valid_std = DataFrame(fold_scores).std().to_dict()
                    valid_std = {f"{k}_std": v for k, v in valid_std.items()}
                else:
                    valid_scores = {}
                    valid_std = {}

                # concatenate the score dicts
                all_scores = dict(**record_path, **record_time, **train_scores,
                                  **valid_scores, **test_scores, **valid_std)
                self.result.append(all_scores)

    def do_experiment(self, train_x, train_y, test_x=None, test_y=None):
        """
        the first call of recurssive fuction.

        Args:
            train_x (pd.DataFrame): training x
            train_y (pd.Series): training y
            test_x (pd.DataFrame): training x
            test_y (pd.Series): training y

        Returns:
            pd.DataFrame: the result
        """
        # clear the results.
        self.result = []

        self.do_stage(train_x, train_y, test_x, test_y, 0, {}, {})
        return self.experiment_results()

    def experiment_results(self, timer=False, std = False) -> DataFrame:
        """
        Args:
            timer (bool): To return the time records.
            std (bool): To return the cv std.

        Returns:
            DataFrame: The experiment results.
        """
        result = DataFrame(self.result)
        to_drop = []
        if not timer:
            to_drop += [i for i in result.columns if i[-5:] == "_time"]
        if not std:
            to_drop += [i for i in result.columns if i[-4:] == "_std"]

        if len(to_drop)==0:
            return result
        else:
            return result.drop(to_drop, axis=1)

    def experiment_predictions(self):
        """
        cv_pred will be empty if cv_result was False in initialization.

        Returns:
            train_pred, cv_pred, test_pred: the prediction of training set, cross validation and  testing set
        """
        return self.train_pred, self.cv_pred, self.test_pred

    def recall_model(self, id):
        """
        query the last experiment result by id and build the pipeline object.

        Todo: A proper way to fit the pipeline object.

        Args:
            id (int): the order of experiment path.

        Returns:
            sklearn.pipeline.Pipeline: ready to use object.
        """

        model_spec = self.result[id]
        model_pipeline = []
        for (step_name, methods) in self.experiment:
            using_method = model_spec[step_name]
            model_pipeline.append((step_name, methods[using_method]))
        return Pipeline(model_pipeline)

    def experiment_detail(self):
        """
        show the experiment settings including:    
            1. models parameters searching range.    

        Returns:
            pandas.DataFrame
        """
        _, models = self.experiment[-1]

        params = []
        for n in list(models):
            m = models[n]
            tmp = m.detail()
            if tmp is not None:
                params.append(tmp)
        return concat(params, axis=0)
