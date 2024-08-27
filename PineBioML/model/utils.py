from PineBioML.report.utils import classification_scores
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
import numpy as np
from pandas import DataFrame

# ToDo: the Integration of .predict and .transform


class Pine():
    """
    Deep first traversal the given experiment setting.    
    the last step of experiment sould be model.    


    note: experiment step and experiment stage is the same thing.
    """

    def __init__(self, experiment):
        """

        Args:
            experiment (list): list of experiment steps. step should be in the form: tuple(str, dict) = (step_name, {method_name: method}). it could be several method in one step and they will fork in deep first traversal.
        """
        self.experiment = experiment
        self.total_stage = len(experiment) - 1
        self.result = []

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
        # if not the last stage
        if stage < self.total_stage:
            # unzip the stage, stage = (stage_name, {operator_name: operator})
            stage_name, operators = self.experiment[stage]

            # fork to next stage according to the diffirent operator (opt)
            for opt_name in operators:
                record[stage_name] = opt_name
                opt = operators[opt_name]

                # transform by operators
                processed_train_x = opt.fit_transform(train_x, train_y)
                if test_x is not None:
                    processed_test_x = opt.transform(test_x)

                # reccursivly call
                self.do_stage(processed_train_x, train_y, processed_test_x,
                              test_y, stage + 1, record)
        # the last layer, it should be models
        else:
            stage_name, operators = self.experiment[stage]
            for opt_name in operators:
                # unzip the stage, stage = (stage_name, {operator_name: operator})
                record[stage_name] = opt_name
                tuner = operators[opt_name]

                # tune/fit the model on training data
                model = tuner.fit(train_x, train_y)
                # compute the training score
                train_scores = classification_scores(model,
                                                     train_x,
                                                     train_y,
                                                     prefix="train_")
                # if there is testing data
                if test_x is not None:
                    # compute the testing score
                    test_scores = classification_scores(model,
                                                        test_x,
                                                        test_y,
                                                        prefix="test_")

                # compute the cross validation score on training set
                valid_scores = {}
                fold_scores = []
                # for each fold
                for (train_idx, valid_idx) in StratifiedKFold(
                        5, shuffle=True,
                        random_state=133).split(train_x, train_y):
                    # fit on fold
                    model.fit(train_x.iloc[train_idx], train_y.iloc[train_idx])
                    # compute fold score
                    fold_scores.append(
                        classification_scores(model,
                                              train_x.iloc[valid_idx],
                                              train_y.iloc[valid_idx],
                                              prefix="cv_"))
                # average the fold scores
                for metric_name in fold_scores[0]:
                    valid_scores[metric_name] = np.array([
                        fold_scores[cv][metric_name] for cv in range(5)
                    ]).mean()

                # concatenate the score dicts
                if test_x is not None:
                    self.result.append(
                        dict(**record, **train_scores, **valid_scores,
                             **test_scores))
                else:
                    self.result.append(
                        dict(**record, **train_scores, **valid_scores))

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

        return DataFrame(self.result)

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
