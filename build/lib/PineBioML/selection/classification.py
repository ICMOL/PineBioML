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

    def __init__(self, k, unbalanced=True, objective="regression"):
        """
        Args:
            unbalanced (bool, optional): False to imply class weight to samples. Defaults to True.
        """
        super().__init__(k=k)

        # parameters
        self.regression = True
        self.unbalanced = unbalanced
        self.name = "LassoLars"

        self.important_path = None

    def reference(self) -> dict[str, str]:
        """
        This function will return reference of this method in python dict.    
        If you want to access it in PineBioML api document, then click on the    >Expand source code     

        Returns:
            dict[str, str]: a dict of reference.
        """
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

        return LassoLars(alpha=C)

    def Scoring(self, x, y=None):
        """
        Using Lasso Lars regression as scoring method.

         Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.

        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.
        
        To do:
            kfold validation performance threshold.
        
        """

        x_train = x.copy()
        #x_train = (x_train - x_train.mean()) / x_train.std()

        y_train = y.copy()
        self.y_encoder = OneHotEncoder(sparse_output=False)
        y_train = self.y_encoder.fit_transform(y_train.to_numpy().reshape(
            -1, 1))
        class_pos_weights = y_train.mean(axis=0)

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
            y = y_train[:, i]
            y = (y - y.mean()) / y.std()
            kernel.fit(x_train, y)

            coef = np.clip(kernel.coef_path_.T, -1, 1)
            alpha = kernel.alphas_
            col = kernel.feature_names_in_

            result.append(pd.DataFrame(coef, columns=col, index=alpha))

        var_dropout_alpha = []
        for s in result:
            tmp = (s != 0).idxmax(axis=0).replace(s.index[0], 0)**2

            # max normalize along sample axis for each output.
            tmp = tmp / tmp.max()
            var_dropout_alpha.append(tmp)

        self.result = result
        self.scores = np.sqrt(
            sum(var_dropout_alpha)).sort_values(ascending=False)
        self.scores.name = self.name
        return self.scores.copy()

    def Plotting(self):
        """
        plot hist graph of selectied feature importance
        """
        fig, ax = plt.subplots(1, 1)
        ax.bar(self.selected_score.index, self.selected_score)
        for label in ax.get_xticklabels(which='major'):
            label.set(rotation=45, horizontalalignment='right')
        ax.set_title(self.name + " score")
        plt.show()

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


class Lasso_bisection_selection(SelectionPipeline):
    """
    Using Lasso (L1 penalty) regression as scoring method.  More specifically, L1 penalty will force feature weights to be zeros. 
    As the coefficient of penalty increases, more and more weights of features got killed and the important feature will remain.

    Lasso_bisection_selection will use binary search to find out when all weights vanish.
    
    The trace of weight vanishment is not support.

    """

    def __init__(self, k, unbalanced=True, objective="regression"):
        """
        Args:
            unbalanced (bool, optional): False to imply class weight to samples. Defaults to True.
            objective (str, optional): one of {"Regression", "BinaryClassification"}
        """
        super().__init__(k=k)
        self.upper_init = 1e+4
        self.lower_init = 1e-4
        self.objective = objective
        if self.objective in ["regression", "Regression"]:
            self.regression = True
        else:
            self.regression = False

        self.sample_weights = 1
        self.unbalanced = unbalanced
        self.name = "LassoBisection"
        self.kernel = self.create_kernel(1)

    def reference(self) -> dict[str, str]:
        """
        This function will return reference of this method in python dict.    
        If you want to access it in PineBioML api document, then click on the    >Expand source code     

        Returns:
            dict[str, str]: a dict of reference.
        """
        refer = super().reference()
        refer[
            self.name() +
            " document"] = "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html"
        refer[
            " publication 1"] = "https://projecteuclid.org/journals/annals-of-statistics/volume-37/issue-5A/High-dimensional-variable-selection/10.1214/08-AOS646.full"
        refer[
            " publication 2"] = "https://www.tandfonline.com/doi/abs/10.1198/016214506000000735?casa_token=5HDhtyCfh40AAAAA:4NxSU97CZubZVpReaQNsSBpqA10_xNhspTQobPnb_z2YXe3Wf-HBHV8OygbqUkmJQPt2Jmp7ZlJPWd0"
        return refer

    def create_kernel(self, C):
        if self.regression:
            return Lasso(alpha=C, warm_start=True)
        else:
            return LogisticRegression(penalty="l1",
                                      C=1 / C,
                                      solver="liblinear",
                                      random_state=142)

    def assign_alpha(self, C, coef=None):
        if self.regression:
            self.kernel.alpha = C
        else:
            self.kernel.C = 1 / C
        if coef is not None:
            self.kernel.coef_ = coef

    def lasso_alive(self, x, y, log_alpha: float, coef=None):
        alpha = self.to_alpha(log_alpha)

        self.assign_alpha(alpha, coef=coef)
        self.kernel.fit(x, y, self.sample_weights)

        importance = self.coef_to_importance(self.kernel.coef_)
        num_alives = (importance > 0).sum()

        return num_alives

    def to_alpha(self, log_alpha):
        return log_alpha  #np.power(10., log_alpha)

    def coef_to_importance(self, coef):
        return np.linalg.norm(coef, ord=2, axis=0)

    def Select(self, x, y):
        """
        Using Lasso (L1 penalty) regression as scoring method.  More specifically, L1 penalty will force feature weights to be zeros. 
        As the coefficient of penalty increases, more and more weights of features got killed and the important feature will remain.

        Lasso_bisection_selection will use binary search to find out when all weights vanish.

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.
            k (int): Number of feature to select. The result may less than k

        Returns:
            pandas.Series: The score for k selected features. May less than k.
        """

        # train test split
        x_train = x.copy()
        y_train = y.copy()

        if self.unbalanced:
            self.sample_weights = compute_sample_weight(
                class_weight="balanced", y=y_train)
        else:
            self.sample_weights = np.ones_like(y_train)

        if self.k == -1:
            self.k = min(x.shape[0], x.shape[1]) // 2

        y_train = OneHotEncoder(sparse_output=False).fit_transform(
            y_train.to_numpy().reshape(-1, 1))

        # Bisection searching
        ### standardize x
        x_train = x_train / x_train.values.std()
        if self.regression:
            y_train = (y_train - y_train.mean(
                axis=0, keepdims=True)) / y_train.std(axis=0, keepdims=True)

        upper = self.upper_init
        upper_alive = self.lasso_alive(x_train, y_train, upper)
        upper_coef = self.kernel.coef_
        #print(upper, upper_alive)

        lower = self.lower_init
        lower_alive = self.lasso_alive(x_train, y_train, lower)
        lower_coef = self.kernel.coef_
        #print(lower, lower_alive)

        alpha = (upper + lower) / 2
        coef = (upper_coef + lower_coef) / 2
        alive = self.lasso_alive(x_train, y_train, alpha, coef)
        counter = 1
        while not alive == self.k:
            #print(alive, alpha)
            if alive >= self.k:
                lower = alpha
                lower_coef = self.kernel.coef_
            else:
                upper = alpha
                upper_coef = self.kernel.coef_

            alpha = (upper + lower) / 2
            coef = (upper_coef + lower_coef) / 2
            alive = self.lasso_alive(x_train, y_train, alpha, coef)

            counter += 1
            if counter > 20:
                break

        self.scores = pd.Series(self.coef_to_importance(self.kernel.coef_),
                                index=x.columns,
                                name=self.name).sort_values(ascending=False)
        self.selected_score = self.scores.head(self.k)
        return self.selected_score


#class Lasso_bisection_selection_V2(Lasso_bisection_selection):


class multi_Lasso_selection(SelectionPipeline):
    """
    A stack of Lasso_bisection_selection. Because of collinearity, if there are a batch of featres with high corelation, only one of them will remain.    
    That leads to diffirent behavior between select k features in a time and select k//n features in n times.    
    """

    def __init__(self, k, objective="regression"):
        """
        Args:
            objective (str, optional): one of {"Regression", "BinaryClassification"}
        """
        super().__init__(k=k)
        self.name = "multi_Lasso"
        self.objective = objective
        self.backend = Lasso_selection  #Lasso_bisection_selection

    def reference(self) -> dict[str, str]:
        """
        This function will return reference of this method in python dict.    
        If you want to access it in PineBioML api document, then click on the    >Expand source code     

        Returns:
            dict[str, str]: a dict of reference.
        """
        refer = super().reference()
        refer[
            " Warning"] = "We do not have a reference and this method's effectivity has not been proven yet."
        return refer

    def Select(self, x, y, n=4):
        """
        Select k//n features for n times, and then concatenate the results.

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.
            n (int, optional): Number of batch which splits k to select. Defaults to 10.

        Returns:
            pandas.Series: The score for k selected features. May less than k.
            
        """
        result = []
        if self.k == -1:
            self.k = min(x.shape[0], x.shape[1]) // 2
        batch_size = self.k // n + 1

        for i in range(n):
            result.append(
                self.backend(k=batch_size,
                             objective=self.objective).Select(x, y))
            x = x.drop(result[-1].index, axis=1)
            if x.shape[1] == 0:
                break
        result = pd.concat(result)
        result = result - result.min()
        result.name = self.name

        self.selected_score = result.sort_values(ascending=False).head(self.k)
        return self.selected_score.copy()


class SVM_selection(SelectionPipeline):
    """
    Using the support vector of linear support vector classifier as scoring method.    

    SVM_selection is scale sensitive in result.    

    <<Feature Ranking Using Linear SVM>> section 3.2

    """

    def __init__(self, k):
        super().__init__(k=k)
        self.kernel = LinearSVC(dual="auto",
                                class_weight="balanced",
                                random_state=142)
        self.name = "SVM"

    def reference(self) -> dict[str, str]:
        """
        This function will return reference of this method in python dict.    
        If you want to access it in PineBioML api document, then click on the    >Expand source code     

        Returns:
            dict[str, str]: a dict of reference.
        """
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

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.

        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.
        """
        self.kernel.fit(x, y)
        svm_weights = np.abs(self.kernel.coef_).sum(axis=0)
        svm_weights /= svm_weights.sum()

        self.scores = pd.Series(svm_weights, index=x.columns,
                                name=self.name).sort_values(ascending=False)
        return self.scores.copy()


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

    def __init__(self, k, bins=10, q=0.05, strategy="c45"):
        """
        Args:
            bins (int, optional): Bins to esimate data distribution entropy. Defaults to 10.
            q (float, optional): Clip data values out of [q, 1-q] percentile to reduce the affect of outliers while estimate entropy. Defaults to 0.05.
            strategy (str, optional): One of {"id3", "c45"}. The strategy to build decision tree. Defaults to "c45".
        """
        super().__init__(k=k)
        self.bins = bins - 1
        self.q = q
        self.strategy = strategy
        self.name = "DT_score_" + self.strategy
        self.label_name = "$%^&UYHGV"

    def reference(self) -> dict[str, str]:
        """
        This function will return reference of this method in python dict.    
        If you want to access it in PineBioML api document, then click on the    >Expand source code     

        Returns:
            dict[str, str]: a dict of reference.
        """
        refer = super().reference()
        refer[self.name() +
              " document"] = "PineBioML.selection.classification.DT_selection"
        refer[
            " publication"] = "Quinlan, J. R. C4.5: Programs for Machine Learning. Morgan Kaufmann Publishers, 1993."
        return refer

    def Scoring(self, x, y=None):
        """
        Using Decision stump (a single Decision tree) to scoring features. Though, single layer stump is equivalent to compare the id3/c4.5 score directly.

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.

        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.
        """
        upper = x.quantile(1 - self.q)
        lower = x.quantile(self.q)
        normed = (x - lower) / (upper - lower + 1e-5)
        normed = normed.clip(0, 1)
        bin_idx = (normed * self.bins - 0.5).round().astype(np.int32)
        columns = bin_idx.columns

        y.name = self.label_name

        #scores = []
        #for i in tqdm(columns):
        #   scores.append(self.compute_col_gini(bin_idx[i], y))
        with parallel_config(backend='loky', n_jobs=-1):
            scores = Parallel()(delayed(self.compute_col_gini)(bin_idx[i], y)
                                for i in tqdm(columns))

        scores = pd.Series(scores, index=columns,
                           name=self.name).sort_values(ascending=False)
        scores = scores - scores.min()
        return scores

    def compute_col_gini(self, col, label):
        i = col.name
        #print(i)
        feature_hists = pd.concat([col, label], axis=1).groupby(i)
        feature_entropy = feature_hists.apply(self.entropy)
        feature_size = feature_hists.apply(len) + 1e-3

        info = (feature_entropy / feature_size).sum()
        gain = 0 - info
        if self.strategy == "c45":
            freq = col.value_counts()
            p = freq / freq.sum()
            split_info = -p * np.log(p) + 1e-6
            gain /= split_info.sum()

        return gain

    def entropy(self, x):
        """
        Estimate entropy

        Args:
            x (pandas.DataFrame): data with bined label

        Returns:
            float: entropy
        """
        label_nums = x[self.label_name].value_counts()
        label_prop = label_nums / label_nums.sum()

        entropy = -(label_prop * np.log(label_prop + 1e-6)).sum()

        return entropy


class RF_selection(SelectionPipeline):
    """
    Using random forest to scoring features by gini/entropy gain.    
    We do not provide permutation importance(VI, variable importance) here.

    """

    def __init__(self, k, unbalanced=True, strategy="gini"):
        """
        Args:
            trees (int, optional): Number of trees. Defaults to 1024*16.
            strategy (str, optional): Scoring strategy, one of {"gini", "entropy"}. Defaults to "gini".
            unbalanced (bool, optional): True to imply class weight to samples. Defaults to True.
        """
        from sklearn.ensemble import RandomForestClassifier
        super().__init__(k=k)
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
                                             min_samples_leaf=3)

        self.name = "RandomForest_" + self.strategy

    def reference(self) -> dict[str, str]:
        """
        This function will return reference of this method in python dict.    
        If you want to access it in PineBioML api document, then click on the    >Expand source code     

        Returns:
            dict[str, str]: a dict of reference.
        """
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

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.

        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.
        """
        self.kernel.n_estimators = round(np.sqrt(x.shape[1]) * 5)
        with parallel_config(backend='loky'):
            self.kernel.fit(x, y)
        score = self.kernel.feature_importances_
        self.scores = pd.Series(score, index=x.columns,
                                name=self.name).sort_values(ascending=False)
        return self.scores.copy()


class XGboost_selection(SelectionPipeline):
    """
    Using XGboost to scoring (gini impurity / entropy) features.

    Warning: If data is too easy, boosting methods is difficult to give score to all features.
    """

    def __init__(self, k, unbalanced=True):
        """
        Args:
            unbalanced (bool, optional): True to imply class weight to samples. Defaults to False.
        """
        super().__init__(k=k)
        self.unbalanced = unbalanced

        from xgboost import XGBClassifier

        self.kernel = XGBClassifier(random_state=142, subsample=0.7)
        self.name = "XGboost"

    def reference(self) -> dict[str, str]:
        """
        This function will return reference of this method in python dict.    
        If you want to access it in PineBioML api document, then click on the    >Expand source code     

        Returns:
            dict[str, str]: a dict of reference.
        """
        refer = super().reference()
        refer[
            self.name() +
            " document"] = "https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor.feature_importances_"
        return refer

    def Scoring(self, x, y=None):
        """
        Using XGboost to scoring (gini impurity / entropy) features.

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.

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
        self.scores = pd.Series(score, index=x.columns,
                                name=self.name).sort_values(ascending=False)
        return self.scores.copy()


class Lightgbm_selection(SelectionPipeline):
    """
    Using Lightgbm to scoring (gini impurity / entropy) features. 

    Warning: If data is too easy, boosting methods is difficult to give score to all features.
    """

    def __init__(self, k, unbalanced=True):
        """
        Args:
            unbalanced (bool, optional): True to imply class weight to samples. Defaults to False.
        """
        super().__init__(k=k)
        self.unbalanced = unbalanced
        from lightgbm import LGBMClassifier

        self.kernel = LGBMClassifier(learning_rate=0.01,
                                     random_state=142,
                                     subsample=0.7,
                                     subsample_freq=1,
                                     verbosity=-1)
        self.name = "Lightgbm"

    def reference(self) -> dict[str, str]:
        """
        This function will return reference of this method in python dict.    
        If you want to access it in PineBioML api document, then click on the    >Expand source code     

        Returns:
            dict[str, str]: a dict of reference.
        """
        refer = super().reference()
        refer[
            self.name() +
            " document"] = "https://lightgbm.readthedocs.io/en/latest/Parameters.html#saved_feature_importance_type"
        return refer

    def Scoring(self, x, y=None):
        """
        Using Lightgbm to scoring (gini impurity / entropy) features.

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.

        Returns:
            pandas.Series or pandas.DataFrame: The score for each feature. Some elements may be empty.
        """
        if self.unbalanced:
            sample_weight = compute_sample_weight(class_weight="balanced", y=y)
        else:
            sample_weight = np.ones(len(y))

        self.kernel.fit(x, y, sample_weight=sample_weight)
        score = self.kernel.feature_importances_
        self.scores = pd.Series(score, index=x.columns,
                                name=self.name).sort_values(ascending=False)
        return self.scores.copy()


class AdaBoost_selection(SelectionPipeline):
    """
    Using AdaBoost to scoring (gini impurity / entropy) features.

    Warning: If data is too easy, boosting methods is difficult to give score to all features.
    """

    def __init__(self, k, unbalanced=True, n_iter=128, learning_rate=0.01):
        """
        Args:
            n_iter (int, optional): Number of trees also number of iteration to boost. Defaults to 64.
            learning_rate (float, optional): boosting learning rate. Defaults to 0.01.
            unbalanced (bool, optional): True to imply class weight to samples. Defaults to False.
        """
        from sklearn.ensemble import AdaBoostClassifier
        super().__init__(k=k)
        self.unbalanced = unbalanced
        self.kernel = AdaBoostClassifier(
            n_estimators=n_iter,
            learning_rate=learning_rate,
            random_state=142,
            algorithm="SAMME",
        )
        self.name = "AdaBoost" + str(n_iter)

    def reference(self) -> dict[str, str]:
        """
        This function will return reference of this method in python dict.    
        If you want to access it in PineBioML api document, then click on the    >Expand source code     

        Returns:
            dict[str, str]: a dict of reference.
        """
        refer = super().reference()
        refer[
            self.name() +
            " document"] = "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier.feature_importances_"
        return refer

    def Scoring(self, x, y=None):
        """
        Using AdaBoost to scoring (gini impurity / entropy) features.

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.

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
        self.scores = pd.Series(score, index=x.columns,
                                name=self.name).sort_values(ascending=False)
        return self.scores.copy()


class ensemble_selector(SelectionPipeline):
    """
    A functional stack of diffirent methods.    
    What we do here is:    
        1. calculate feature importance in different methods.    
        2. standardize the scores and then averaging through methods.    
        3. If z_importance_threshold not None, then all features with averaging score higher than z_importance_threshold will be selected.    
           else top k feature with averaging score will be selected.    
    """

    def __init__(self, k=-1, z_importance_threshold: int = None):
        """

        Args:
            k (int, optional): The number of features to be selected. Defaults to -1.
            RF_trees (int, optional): number of trees using for randomforest. Defaults to 1024.
            z_importance_threshold (int, optional): The threshold to picking features. Defaults to None.
        
        TODO:
            auto adjust the RF_Trees by number of input features.
        """
        self.k = k
        self.z_importance_threshold = z_importance_threshold

        self.kernels = {
            "c45": DT_selection(k=k, strategy="c45"),
            "RF_gini": RF_selection(k=k, strategy="gini"),
            "Lasso": Lasso_selection(k=k),
            "multi_Lasso": multi_Lasso_selection(k=k),
            "SVM": SVM_selection(k=k),

            #"AdaBoost": AdaBoost_selection(k=k),
            #"XGboost": XGboost_selection(k=k),
            #"Lightgbm": Lightgbm_selection(k=k)
        }

    def reference(self) -> dict[str, str]:
        """
        This function will return reference of this method in python dict.    
        If you want to access it in PineBioML api document, then click on the    >Expand source code     

        Returns:
            dict[str, str]: a dict of reference.
        """
        refer = super().reference()
        refer[
            self.name() +
            " WARNING"] = "This method has no reference yet. That means the effectivity has not been proven yet. It somehow works in experience."
        return refer

    def Select(self, x, y):
        """
        Calling all the methods in kernel sequancially.

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.
            k (int): Number of feature to select. The result may less than k

        Returns:
            pandas.Series: The concatenated results. Top k (may less than k) important feature from diffient methods.
        """
        results = []
        for method in self.kernels:
            print("Using ", method, " to select.")
            start_time = time.time()
            results.append(self.kernels[method].Select(x.copy(), y))
            end_time = time.time()
            print(method,
                  " is done. Using {t:.4f}\n".format(t=end_time - start_time))

        self.selected_score = pd.concat(results, axis=1)
        return self.selected_score

    def fit(self, x, y):
        """
        sklearn api

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods.
        """
        importance = self.Select(x, y)

        return self

    def transform(self, x):
        z_scores = (self.selected_score - self.selected_score.mean()) / (
            self.selected_score.std() + 1e-4)

        scores = z_scores.sum(axis=1)

        z_scores = (scores - scores.mean()) / (scores.std() + 1e-4)
        z_scores = z_scores.sort_values(ascending=False)

        if self.z_importance_threshold is None:
            return x[z_scores.index[:self.k]]
        else:
            return x[z_scores[z_scores > self.z_importance_threshold].index]

    def fit_transform(self, x, y):
        self.fit(x, y)
        return self.transform(x)

    def plotting(self):
        for method in self.kernels:
            self.kernels[method].plotting()
