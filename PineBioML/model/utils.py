from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from pandas import DataFrame, Series, concat


class sklearn_esitimator_wrapper():

    def __init__(self, kernel):
        self.kernel = kernel

    def fit(self, x, y):
        self.label_name = y.name
        self.kernel.fit(x, y)
        return self

    def predict(self, x):
        return Series(self.kernel.predict(x),
                      index=x.index,
                      name=self.label_name)

    def predict_proba(self, x):
        if "predict_proba" in dir(self.kernel):
            return DataFrame(self.kernel.predict_proba(x),
                             index=x.index,
                             columns=self.kernel.classes_)
        else:
            raise NotImplementedError(
                "{} do not have attribute 'predict_proba'.".format(
                    self.kernel.__str__()))


class classification_scorer():

    def __init__(self, target_label=None, prefix="", multi_class_extra=False):
        self.prefix = prefix
        self.target_label = target_label
        self.multi_class_extra = multi_class_extra

    def score(self, y_true, y_pred_prob):
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

    def __init__(self, prefix="", target_label=None):
        self.prefix = prefix

    def score(self, y_true, y_pred):

        result = {}
        result["rmse"] = metrics.root_mean_squared_error(y_true, y_pred)
        result["r2"] = metrics.r2_score(y_true, y_pred)
        result["mape"] = metrics.mean_absolute_percentage_error(y_true, y_pred)
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


    note: experiment step and experiment stage is the same thing.
    """

    def __init__(self, experiment, target_label=None, cv_result=False):
        """

        Args:
            experiment (list): list of experiment steps. step should be in the form: tuple(str, dict) = (step_name, {method_name: method}). it could be several method in one step and they will fork in deep first traversal.
        """
        self.experiment = experiment
        self.total_stage = len(experiment)
        self.result = []
        self.train_pred = []
        self.cv_pred = []
        self.test_pred = []
        self.target_label = target_label
        self.cv_result = cv_result

    def do_stage(self, train_x, train_y, test_x, test_y, stage, record):
        """
        the recursive function to traversal the experiment setting.    
        the socres and path will be stored in self.result, so there is no return in recursive function.     

        Args:
            train_x (pd.DataFrame): training x
            train_y (pd.Series): training y
            test_x (pd.DataFrame): training x
            test_y (pd.Series): training y
            stage (int): the order of current stage in the experiment setting
            record (dict): record the traversal path in a dict of str
        """
        # unzip the stage, stage = (stage_name, {operator_name: operator})
        stage_name, operators = self.experiment[stage]

        # fork to next stage according to the diffirent operator (opt)
        for opt_name in operators:
            record[stage_name] = opt_name
            opt = operators[opt_name]

            # if not the last stage
            if stage < self.total_stage - 1:
                # transform by operators
                processed_train_x = opt.fit_transform(train_x, train_y)
                if test_x is not None:
                    processed_test_x = opt.transform(test_x)

                # reccursivly call
                self.do_stage(processed_train_x, train_y, processed_test_x,
                              test_y, stage + 1, record)

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

                # tune/fit the model on training data
                model.fit(train_x, train_y)

                # compute the training score
                train_pred = f(train_x)
                self.train_pred.append(train_pred)
                train_scores = scorer(prefix="train_",
                                      target_label=self.target_label).score(
                                          train_y, train_pred)

                if test_x is not None:
                    # if there is testing data, compute the testing score.
                    test_pred = f(test_x)
                    self.test_pred.append(test_pred)
                    test_scores = scorer(prefix="test_",
                                         target_label=self.target_label).score(
                                             test_y, test_pred)
                else:
                    test_scores = {}

                if self.cv_result:
                    # compute the cross validation score on training set
                    fold_scores = []
                    cv_pred = []
                    cross_validation = StratifiedKFold(n_splits=5,
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
                        cv_pred.append(fold_pred)
                        fold_scores.append(
                            scorer(prefix="cv_",
                                   target_label=self.target_label).score(
                                       train_y.iloc[valid_idx], fold_pred))
                    # average the fold scores
                    """
                    valid_scores = {}
                    for metric_name in fold_scores[0].keys():
                        valid_scores[metric_name] = np.array([
                            fold_scores[cv][metric_name] for cv in range(5)
                        ]).mean()
                    """
                    self.cv_pred.append(concat(cv_pred, axis=0))
                    valid_scores = DataFrame(fold_scores).mean().to_dict()
                else:
                    valid_scores = {}

                # concatenate the score dicts
                all_scores = dict(**record, **train_scores, **valid_scores,
                                  **test_scores)
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

        self.do_stage(train_x, train_y, test_x, test_y, 0, {})
        return self.experiment_results()

    def experiment_results(self):
        return DataFrame(self.result)

    def experiment_predictions(self):
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
