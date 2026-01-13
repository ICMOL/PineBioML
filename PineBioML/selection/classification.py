from . import SelectionPipeline

import pandas as pd
import numpy as np
from tqdm import tqdm
import time

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable

from joblib import Parallel, parallel_config, delayed

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Lasso, LassoLars, LogisticRegression
from sklearn.svm import LinearSVC


class Lasso_selection(SelectionPipeline):
    """
    Using Lasso (L1 penalty) regression as scoring method. L1 penalty will force feature weights to be zeros.    
    As the penalty increases, more and more regression coefficients vanish and the ones coresponding to important variables will remain.    

    The Lasso_selection is base on [Lasso Lars](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html).    

    Currently, this do not support Lasso + logistic regression, so:    
        Binary classification problem will be regarded as a regression to {0, 1}.    
        Multi-class classification problem will be devided into ovr (one vs rest) classification and the score of features over ovr will be combined by average weighted class weight.    

    ~~Lasso_selection will use grid search to find out when all weights vanish.    ~~
    """

    def __init__(self,
                 k=None,
                 z_importance_threshold=1.,
                 unbalanced=True,
                 n_cv=5):
        """
        Args:
            unbalanced (bool, optional): False to imply class weight to samples. Defaults to True.
        """
        super().__init__(k=k,
                         z_importance_threshold=z_importance_threshold,
                         n_cv=n_cv)

        # parameters
        self.regression = True
        self.unbalanced = unbalanced
        self.name = "LassoLars"

    def reference(self) -> dict[str, str]:
        refer = super().reference()
        refer[
            self.name() +
            " document"] = "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html"
        refer[
            " publication Lasso 1"] = "https://projecteuclid.org/journals/annals-of-statistics/volume-37/issue-5A/High-dimensional-variable-selection/10.1214/08-AOS646.full"
        refer[
            " publication Lasso 2"] = "https://www.tandfonline.com/doi/abs/10.1198/016214506000000735?casa_token=5HDhtyCfh40AAAAA:4NxSU97CZubZVpReaQNsSBpqA10_xNhspTQobPnb_z2YXe3Wf-HBHV8OygbqUkmJQPt2Jmp7ZlJPWd0"
        refer[" publication Lars"] = "https://arxiv.org/pdf/math/0406456"

        return refer

    def create_kernel(self, C=1e-2):
        """
        Create diffirent kernel according to opjective.

        Args:
            C (float): The coefficient to L1 penalty.

        Returns:
            sklearn.linearmodel: a kernel of sklearn linearmodel
        
        TODO:
            1. auto C tuner or adapter.
            
        """

        return LassoLars(alpha=C, random_state=142)

    def Scoring(self, x, y=None):
        """
        Using Lasso Lars regression as scoring method.
                
        """

        x_train = x.copy()
        y_train = y.copy()

        self.y_encoder = OneHotEncoder(sparse_output=False)
        self.y_encoder.fit(y.to_numpy().reshape(-1, 1))
        y_train = self.y_encoder.transform(y_train.to_numpy().reshape(-1, 1))

        kernel = self.create_kernel()

        ### TODO: Sample weight for Lars.
        if y_train.shape[1] == 2:
            # Binary classification
            task_set = [1]
        else:
            # multi-class classification
            ### TODO: Multinomial classfication for Lars.
            task_set = range(y_train.shape[1])

        result = []
        for i in task_set:
            # TODO: cross validation, alpha interpolation
            y = y_train[:, i]
            y = (y - y.mean()) / y.std()
            kernel.fit(x_train, y)

            coef = np.clip(kernel.coef_path_.T, -1, 1)
            alpha = kernel.alphas_
            col = kernel.feature_names_in_

            result.append(pd.DataFrame(coef, columns=col, index=alpha))

        var_dropout_alpha = []
        for s in result:
            # the right end side(alpha) of the range that variable has a non-zero lars coefficient.
            tmp = (s != 0).idxmax(axis=0).replace(s.index[0], 0)**2

            # normalizing each task.
            tmp = tmp / tmp.max()
            var_dropout_alpha.append(tmp)

        # collect the results
        self.result = result
        scores = np.sqrt(sum(var_dropout_alpha)).sort_values(ascending=False)
        scores.name = self.name
        return scores

    def Plotting(self):
        super().Plotting()

        global_selected = self.selected_score.index
        for i_th in range(len(self.result)):
            s = self.result[-i_th - 1]

            # get the features ever alive.
            alive_col = s.columns[(s != 0).any(axis=0)]

            # show the label of the top-5 long-lived features
            noting_col = (s != 0).idxmax(axis=0).replace(
                s.index[0], 0).sort_values().tail(5).index

            plt_cmap = plt.get_cmap("Blues")
            plt_norm = plt.Normalize(vmin=0, vmax=s.index[1])
            fig, ax = plt.subplots(layout='constrained')

            for col in alive_col:
                var_coef = s[col]
                self.plot_coef_1var(
                    alpha=var_coef.index,
                    coef=var_coef.values,
                    global_selected=col in global_selected,
                    coef_name=col if col in noting_col else None,
                    cmap=plt_cmap,
                    norm=plt_norm,
                )

            plt.axhline(y=0, color="grey", linestyle="--")
            plt.xlabel("alpha")
            plt.ylabel("coefficient")
            plt.title("class-{} Lasso".format(
                self.y_encoder.categories_[0][i_th]))

            scalar_mappable = ScalarMappable(norm=plt_norm, cmap=plt_cmap)
            fig.colorbar(scalar_mappable,
                         ax=ax,
                         orientation='vertical',
                         label='Variable dropout alpha')
            plt.xscale('log')
            legend_elements = [
                Line2D([0], [0],
                       color="b",
                       label="Globally important",
                       linestyle="solid"),
                Line2D([0], [0],
                       color="b",
                       label="Partially important",
                       linestyle="--"),
            ]
            ax.legend(handles=legend_elements, loc="upper right")
            plt.show()

    def plot_coef_1var(self, alpha, coef, global_selected, coef_name, cmap,
                       norm):
        # plot the coef curve
        linestyle = "solid" if global_selected else "--"

        alive = coef != 0
        alive[alive.argmax() - 1] = True

        alpha = alpha[alive]
        coef = coef[alive]

        plt.plot(alpha,
                 coef,
                 label=coef_name,
                 c=cmap(norm(alpha.max())),
                 linestyle=linestyle)

        # plot the annotate
        if coef_name is not None:
            annotate_idx = np.abs(coef).argmax()
            annotate_coor = (alpha[annotate_idx], coef[annotate_idx])
            text_coor = (alpha[annotate_idx], coef[annotate_idx])
            plt.annotate(
                coef_name,
                xy=annotate_coor,
                xytext=text_coor,
            )


class multi_Lasso_selection(SelectionPipeline):
    """
    A stack of Lasso_selection. Because of collinearity, if there are a batch of featres with high corelation, only one of them will remain.    
    That leads to diffirent behavior between select k features in a time and select k//n features in n times.    
    """

    def __init__(self, k=None, z_importance_threshold=1., n=4, n_cv=5):
        """
        Args:
            n (int , optional): How many times should lasso be performed.    
            objective (str, optional): one of {"Regression", "BinaryClassification"}
        """
        super().__init__(k=k,
                         z_importance_threshold=z_importance_threshold,
                         n_cv=n_cv)
        self.name = "multi_Lasso"
        self.backend = Lasso_selection  #Lasso_bisection_selection
        self.n = n
        self.n_cv = 1  # multi lasso calls lasso which already applied cv

    def reference(self) -> dict[str, str]:
        refer = super().reference()
        refer[
            " Warning"] = "We do not have a reference and this method's effectivity has not been proven yet."
        return refer

    def Scoring(self, x, y):
        result = []
        if self.k == -1 or self.k is None:
            self.k = min(x.shape[0], x.shape[1]) // 2
        batch_size = self.k // self.n + 1

        num_selected = 0
        #for i in range(self.n):
        while (num_selected < self.k):
            kernel = self.backend(k=batch_size).fit(x, y)
            result.append(kernel.selected_score)
            batch_selected = result[-1].index
            x = x.drop(batch_selected, axis=1)
            num_selected += len(batch_selected)
            if x.shape[1] == 0:
                break
        result = pd.concat(result).sort_values(ascending=False)
        #result = result - result.min()
        result.name = self.name

        return result


class SVM_selection(SelectionPipeline):
    """
    Using the support vector of linear support vector classifier as scoring method.    

    SVM_selection is scale sensitive in result.    

    <<Feature Ranking Using Linear SVM>> section 3.2

    """

    def __init__(self, k=None, z_importance_threshold=1., n_cv=5):
        super().__init__(k=k,
                         z_importance_threshold=z_importance_threshold,
                         n_cv=n_cv)
        self.kernel = LinearSVC(dual="auto",
                                class_weight="balanced",
                                random_state=142)
        self.name = "SVM"

    def reference(self) -> dict[str, str]:
        refer = super().reference()
        refer[
            self.name() +
            " document"] = "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html"
        refer[
            " publication - section 3.2"] = "https://www.csie.ntu.edu.tw/~cjlin/papers/causality.pdf"
        return refer

    def Scoring(self, x, y=None):
        """
        Using the support vector of linear support vector classifier as scoring method.

        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.
        """
        self.kernel.fit(x, y)
        svm_weights = np.abs(self.kernel.coef_).sum(axis=0)
        svm_weights /= svm_weights.sum()

        scores = pd.Series(svm_weights, index=x.columns,
                           name=self.name).sort_values(ascending=False)
        return scores


class DT_selection(SelectionPipeline):
    """
    A child class of SelectionPipeline.

    Using Decision stump (a single Decision tree) to scoring features.    
    What we do here is:    
        1. normalize data    
        2. transform data into frequency domain by binning through a certain column and applying value_counts    
        3. estimate entropy    
        4. compute c4.5    
    """

    def __init__(self,
                 k=None,
                 z_importance_threshold=1.,
                 bins=10,
                 q=0.05,
                 strategy="c45",
                 n_cv=5):
        """
        Args:
            bins (int, optional): Bins to esimate data distribution entropy. Defaults to 10.
            q (float, optional): Clip data values out of [q, 1-q] percentile to reduce the affect of outliers while estimate entropy. Defaults to 0.05.
            strategy (str, optional): One of {"gini", "c45"}. The strategy to build decision tree. Defaults to "c45".
        """
        super().__init__(k=k,
                         z_importance_threshold=z_importance_threshold,
                         n_cv=n_cv)
        self.bins = bins - 1
        self.q = q
        self.strategy = strategy
        self.name = "DT_score_" + self.strategy

    def reference(self) -> dict[str, str]:
        refer = super().reference()
        refer[self.name() +
              " document"] = "PineBioML.selection.classification.DT_selection"
        refer[
            " publication"] = "Quinlan, J. R. C4.5: Programs for Machine Learning. Morgan Kaufmann Publishers, 1993."
        return refer

    def Scoring(self, x, y=None):
        """
        Using Decision stump (a single Decision tree) to scoring features. Though, single layer stump is equivalent to compare the id3/c4.5 score directly.

        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.
        """
        self.y_encoder = OneHotEncoder(sparse_output=False)
        y = self.y_encoder.fit_transform(y.to_numpy().reshape(
            -1, 1)).argmax(axis=-1)

        if self.strategy == 'c45':
            score = self.best_splits_c45_batch_optimized(x, y)
        elif self.strategy == 'gini':
            score = self.best_splits_gini_batch_optimized(x, y)

        scores = pd.Series(score, index=x.columns,
                           name=self.name).sort_values(ascending=False)
        return scores

    def compute_gain_ratio(self, y_sorted, n_classes, n, eps=1e-12):
        """
        Compute best Gain Ratio for one sorted feature
        """
        # one-hot encode
        one_hot = np.eye(n_classes)[y_sorted]

        # cumulative counts (exclude last split)
        left_counts = np.cumsum(one_hot, axis=0)[:-1]
        total_counts = left_counts[-1]
        right_counts = total_counts - left_counts

        # sample counts
        left_totals = np.arange(1, n)
        right_totals = n - left_totals

        # ---- entropy computation ----
        def entropy(counts, totals):
            probs = counts / totals[:, None]
            probs = np.clip(probs, eps, 1.0)
            return -np.sum(probs * np.log2(probs), axis=1)

        # parent entropy
        parent_probs = total_counts / n
        parent_probs = np.clip(parent_probs, eps, 1.0)
        parent_entropy = -np.sum(parent_probs * np.log2(parent_probs))

        # left / right entropy
        entropy_left = entropy(left_counts, left_totals)
        entropy_right = entropy(right_counts, right_totals)

        # information gain
        info_gain = parent_entropy - (
            (left_totals / n) * entropy_left +
            (right_totals / n) * entropy_right
        )

        # split information
        p_left = left_totals / n
        p_right = right_totals / n
        split_info = -(
            p_left * np.log2(np.clip(p_left, eps, 1.0)) +
            p_right * np.log2(np.clip(p_right, eps, 1.0))
        )

        # gain ratio
        gain_ratio = info_gain / np.clip(split_info, eps, None)

        # C4.5 chooses the **maximum** gain ratio
        return np.max(gain_ratio)

    def compute_gini(self, y_sorted, n_classes, n):
        one_hot = np.eye(n_classes)[y_sorted]
        left_counts = np.cumsum(one_hot, axis=0)[:-1]

        # right counts
        total_counts = left_counts[-1]
        right_counts = total_counts - left_counts

        # total counts
        left_totals = np.arange(1, n)
        right_totals = n - left_totals

        # vectorizing left & right Gini computing
        gini_left = 1.0 - np.sum(
            (left_counts / left_totals[:, np.newaxis])**2, axis=1)
        gini_right = 1.0 - np.sum(
            (right_counts / right_totals[:, np.newaxis])**2, axis=1)

        # weighted Gini
        weighted_gini = (left_totals * gini_left +
                         right_totals * gini_right) / n
        return np.min(weighted_gini)

    def best_splits_gini_batch_optimized(self, X, Y):
        """
        compute gini in batch
        
        Args:
            X (pandas.DataFrame): (n_samples, n_features) in numerical dtypes.
            y (pandas.Series): (n_samples,) in int.
        
        return:
            best_ginis: (n_features,)
            best_thresholds: (n_features,) placeholder, we don't need it here
        """
        x = X  #.to_numpy()
        y = Y  #.to_numpy().astype(np.int8)
        n, n_features = X.shape
        n_classes = y.max() + 1

        # sorting all the features
        sort_idx = np.argsort(x, axis=0)

        #X_sorted = np.take_along_axis(X, sort_idx, axis=0)
        y_sorted_all = np.take_along_axis(np.tile(y[:, None], (1, n_features)),
                                          sort_idx,
                                          axis=0)

        best_ginis = np.ones(n_features)

        with parallel_config(backend='loky', n_jobs=-1):
            best_ginis = Parallel()(
                delayed(self.compute_gini)(y_sorted_all[:, i], n_classes, n)
                for i in tqdm(np.arange(n_features)))

        return best_ginis
    
    def best_splits_c45_batch_optimized(self, X, Y):
        """
        compute C4.5 gain ratio in batch
        
        Args:
            X (pandas.DataFrame): (n_samples, n_features)
            Y (pandas.Series): (n_samples,) int labels
        
        return:
            best_gain_ratios: (n_features,)
        """
        x = X
        y = Y
        n, n_features = X.shape
        n_classes = y.max() + 1

        # sort indices per feature
        sort_idx = np.argsort(x, axis=0)

        # reorder y for each feature
        y_sorted_all = np.take_along_axis(
            np.tile(y[:, None], (1, n_features)),
            sort_idx,
            axis=0
        )

        with parallel_config(backend='loky', n_jobs=-1):
            best_gain_ratios = Parallel()(
                delayed(self.compute_gain_ratio)(
                    y_sorted_all[:, i], n_classes, n
                )
                for i in tqdm(range(n_features))
            )

        return np.array(best_gain_ratios)

class RF_selection(SelectionPipeline):
    """
    Using random forest to scoring features by gini/entropy gain.    
    We do not provide permutation importance(VI, variable importance) here.

    """

    def __init__(self,
                 k=None,
                 z_importance_threshold=1.,
                 unbalanced=True,
                 strategy="gini",
                 n_cv=5):
        """
        Args:
            strategy (str, optional): Scoring strategy, one of {"gini", "entropy"}. Defaults to "gini".
            unbalanced (bool, optional): True to imply class weight to samples. Defaults to True.
        """
        from sklearn.ensemble import RandomForestClassifier
        super().__init__(k=k,
                         z_importance_threshold=z_importance_threshold,
                         n_cv=n_cv)
        self.strategy = strategy
        if unbalanced:
            class_weight = "balanced"
        else:
            class_weight = None

        self.kernel = RandomForestClassifier(n_estimators=100,
                                             n_jobs=-1,
                                             max_samples=0.7,
                                             class_weight=class_weight,
                                             criterion=strategy,
                                             verbose=0,
                                             random_state=142,
                                             min_samples_leaf=2)

        self.name = "RandomForest_" + self.strategy

    def reference(self) -> dict[str, str]:

        refer = super().reference()
        refer[self.name() + " document"] = ""
        refer[
            " publication cons"] = "https://link.springer.com/article/10.1186/1471-2105-8-25"
        refer[
            " publication pros 1"] = "https://link.springer.com/article/10.1186/1471-2105-10-213"
        refer[
            " publication pros 2"] = "https://www.sciencedirect.com/science/article/pii/S0167947307003076"
        refer[
            " publication survey"] = "https://www.cs.cmu.edu/~qyj/papersA08/11-rfbook.pdf"

        return refer

    def Scoring(self, x, y=None):
        """
        Using random forest to scoring (gini impurity / entropy) features.

        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.
        """
        self.kernel.n_estimators = round(2 * np.sqrt(x.shape[1]) *
                                         np.log(x.shape[1]))
        with parallel_config(backend='loky'):
            self.kernel.fit(x, y)
        score = self.kernel.feature_importances_
        scores = pd.Series(score, index=x.columns,
                           name=self.name).sort_values(ascending=False)
        return scores


class XGboost_selection(SelectionPipeline):
    """
    Using XGboost to scoring (gini impurity / entropy) features.

    Warning: If data is too easy, boosting methods is difficult to give score to all features.
    """

    def __init__(self,
                 k=None,
                 z_importance_threshold=1.,
                 unbalanced=True,
                 n_cv=5):
        """
        Args:
            unbalanced (bool, optional): True to imply class weight to samples. Defaults to False.
        """
        super().__init__(k=k,
                         z_importance_threshold=z_importance_threshold,
                         n_cv=n_cv)
        self.unbalanced = unbalanced

        from xgboost import XGBClassifier

        self.kernel = XGBClassifier(random_state=142, subsample=0.7)
        self.name = "XGboost"

    def reference(self) -> dict[str, str]:
        refer = super().reference()
        refer[
            self.name() +
            " document"] = "https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor.feature_importances_"
        return refer

    def Scoring(self, x, y=None):
        """
        Using XGboost to scoring (gini impurity / entropy) features.

        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.
        """
        if self.unbalanced:
            sample_weight = compute_sample_weight(class_weight="balanced", y=y)
        else:
            sample_weight = np.ones(len(y))

        y = OneHotEncoder(sparse_output=False).fit_transform(
            y.to_numpy().reshape(-1, 1))
        self.kernel.fit(x, y, sample_weight=sample_weight)
        score = self.kernel.feature_importances_
        scores = pd.Series(score, index=x.columns,
                           name=self.name).sort_values(ascending=False)
        return scores


class Lightgbm_selection(SelectionPipeline):
    """
    Using Lightgbm to scoring (gini impurity / entropy) features. 

    Warning: If data is too easy, boosting methods is difficult to give score to all features.
    """

    def __init__(self,
                 k=None,
                 z_importance_threshold=1.,
                 unbalanced=True,
                 n_cv=5):
        """
        Args:
            unbalanced (bool, optional): True to imply class weight to samples. Defaults to False.
        """
        super().__init__(k=k,
                         z_importance_threshold=z_importance_threshold,
                         n_cv=n_cv)
        self.unbalanced = unbalanced
        from lightgbm import LGBMClassifier

        self.kernel = LGBMClassifier(learning_rate=0.01,
                                     random_state=142,
                                     subsample=0.7,
                                     subsample_freq=1,
                                     verbosity=-1)
        self.name = "Lightgbm"

    def reference(self) -> dict[str, str]:
        refer = super().reference()
        refer[
            self.name() +
            " document"] = "https://lightgbm.readthedocs.io/en/latest/Parameters.html#saved_feature_importance_type"
        return refer

    def Scoring(self, x, y=None):
        """
        Using Lightgbm to scoring (gini impurity / entropy) features.

        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.
        """
        if self.unbalanced:
            sample_weight = compute_sample_weight(class_weight="balanced", y=y)
        else:
            sample_weight = np.ones(len(y))

        self.kernel.fit(x, y, sample_weight=sample_weight)
        score = self.kernel.feature_importances_
        scores = pd.Series(score, index=x.columns,
                           name=self.name).sort_values(ascending=False)
        return scores


class AdaBoost_selection(SelectionPipeline):
    """
    Using AdaBoost to scoring (gini impurity / entropy) features.

    Warning: If data is too easy, boosting methods is difficult to give score to all features.
    """

    def __init__(self,
                 k=None,
                 z_importance_threshold=1.,
                 unbalanced=True,
                 n_iter=128,
                 learning_rate=0.01,
                 n_cv=5):
        """
        Args:
            n_iter (int, optional): Number of trees also number of iteration to boost. Defaults to 64.
            learning_rate (float, optional): boosting learning rate. Defaults to 0.01.
            unbalanced (bool, optional): True to imply class weight to samples. Defaults to False.
        """
        from sklearn.ensemble import AdaBoostClassifier
        super().__init__(k=k,
                         z_importance_threshold=z_importance_threshold,
                         n_cv=n_cv)
        self.unbalanced = unbalanced
        self.kernel = AdaBoostClassifier(
            n_estimators=n_iter,
            learning_rate=learning_rate,
            random_state=142,
        )
        self.name = "AdaBoost" + str(n_iter)

    def reference(self) -> dict[str, str]:

        refer = super().reference()
        refer[
            self.name() +
            " document"] = "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier.feature_importances_"
        return refer

    def Scoring(self, x, y=None):
        """
        Using AdaBoost to scoring (gini impurity / entropy) features.

        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.
        """
        print("I don't have a progress bar but I am running")
        if self.unbalanced:
            sample_weight = compute_sample_weight(class_weight="balanced", y=y)
        else:
            sample_weight = np.ones(len(y))
        self.kernel.fit(x, y, sample_weight)
        score = self.kernel.feature_importances_
        scores = pd.Series(score, index=x.columns,
                           name=self.name).sort_values(ascending=False)
        return scores


class ensemble_selector(SelectionPipeline):
    """
    A functional stack of diffirent methods.    
    What we do here is:    
        1. calculate feature importance in different methods.    
        2. standardize the scores and then averaging through methods.    
        3. If z_importance_threshold not None, then all features with averaging score higher than z_importance_threshold will be selected.    
           else top k feature with averaging score will be selected.    
    """

    def __init__(self,
                 k: int = None,
                 z_importance_threshold: float = 1.,
                 n_cv=5):

        super().__init__(k=k,
                         z_importance_threshold=z_importance_threshold,
                         n_cv=n_cv)
        self.name = "ensemble"
        self.kernels = {
            "c45": DT_selection(k=k, strategy="c45", n_cv=n_cv),
            "RF_gini": RF_selection(k=k, strategy="gini", n_cv=n_cv),
            "Lasso": Lasso_selection(k=k, n_cv=n_cv),
            "multi_Lasso": multi_Lasso_selection(k=k, n_cv=n_cv),
            "SVM": SVM_selection(k=k, n_cv=n_cv),

            #"AdaBoost": AdaBoost_selection(k=k),
            #"XGboost": XGboost_selection(k=k),
            #"Lightgbm": Lightgbm_selection(k=k)
        }
        self.n_cv = 1

    def reference(self) -> dict[str, str]:
        refer = super().reference()
        refer[
            self.name() +
            " WARNING"] = "This method has no reference yet. That means the effectivity has not been proven yet. It somehow works in experience."
        return refer

    def Scoring(self, x, y=None):
        results = []
        for method in self.kernels:
            print("Using ", method, " to score.")
            start_time = time.time()

            self.kernels[method].fit(x, y)

            results.append(self.kernels[method].selected_score)
            end_time = time.time()
            print(method,
                  " is done. Using {t:.4f}\n".format(t=end_time - start_time))

        scores = pd.concat(results, axis=1)
        z_scores = (scores - scores.mean()) / (scores.std() + 1e-4)
        scores[self.name] = z_scores.sum(axis=1)

        return scores

    def Select(self, scores):
        z_scores = scores[self.name].sort_values(ascending=False)

        return super().Select(z_scores)

    def what_matters(self):
        return self.scores

    def Plotting(self):
        for method in self.kernels:
            self.kernels[method].Plotting()
