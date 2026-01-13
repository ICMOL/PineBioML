from abc import ABC, abstractmethod
from typing import Union

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from seaborn import scatterplot, heatmap, pairplot, color_palette

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import OneHotEncoder
import sklearn.metrics as metrics


class basic_plot(ABC):
    """
    the base class of plots.    
    Using make_figure(x, y) to generate a figure.    

    """

    def __init__(self,
                 prefix: str = "",
                 save_path: str = "./",
                 save_fig: bool = False,
                 show_fig: bool = True):
        """

        Args:
            prefix (str, optional): the describe or title for the plot. the prefix will be added into the title of plot and saving name. Defaults to "".
            save_path (str, optional): the path to export the figure. Defaults to "./".
            save_fig (bool, optional): whether to export the figure or not. Defaults to False.
            show_fig (bool, optional): whether to show the figure or not. Defaults to True.
        """

        self.prefix = prefix
        self.save_path = save_path
        self.save_fig = save_fig
        self.show_fig = show_fig

    def save_name(self):
        """
        the file name to the saving figure.    

        will be "{prefix} {name}.png"
        """

        return "{} {}".format(self.prefix, self.name)

    def reference(self) -> dict[str, str]:
        """
        This function will return reference of this method in python dict.    
        If you want to access it in PineBioML api document, then click on the    >Expand source code     

        Returns:
            dict[str, str]: a dict of reference.
        """
        refer = {
            "matplotlib document": "https://matplotlib.org/",
            "seaborn document": "https://seaborn.pydata.org/"
        }

        return refer

    @abstractmethod
    def draw(self, x: pd.DataFrame, y: pd.Series = None):
        """
        How and what to draw should be implemented in here.    

        Args:
            x (pd.DataFrame): feature
            y (pd.Series, optional): label. Defaults to None.
        """
        pass

    def make_figure(self, x: pd.DataFrame, y: pd.Series = None):
        """
        1. draw(x, y)    
        2. To save or to show the result.

        Args:
            x (pd.DataFrame): features
            y (pd.Series, optional): label. Defaults to None.
        """

        self.draw(x, y)

        if self.save_fig:
            plt.savefig(self.save_path + self.save_name(), bbox_inches='tight')
        if self.show_fig:
            plt.show()
        else:
            plt.clf()


class pca_plot(basic_plot):
    """
    Calling pca_plot().make_figure(x) or pca_plot().make_figure(x, y) to draw a pca plot of x with n_pc components.    

    """

    def __init__(self,
                 n_pc: int = 4,
                 discrete_legend: bool = True,
                 prefix: str = "",
                 save_path: str = "./output/images/",
                 save_fig: bool = True,
                 show_fig: bool = True):
        """

        Args:
            n_pc (int, optional): number of precipal compoment to plot. Defaults to 4.
            discrete_legend (bool, optional): To color the plot based on y in discrete hue or continuous color bar. If y is continuous, then you should set it to False. Defaults to True.
        """
        super().__init__(prefix=prefix,
                         save_path=save_path,
                         save_fig=save_fig,
                         show_fig=show_fig)
        self.n_pc = n_pc
        self.discrete_legend = discrete_legend
        self.name = "PCA plot"

    def reference(self) -> dict[str, str]:

        refer = super().reference()
        refer[
            self.name +
            " document"] = "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html"

        return refer

    def draw(self, x: pd.DataFrame, y: pd.Series = None):
        """
        Using x to draw a pca plot. The difference between pca_plot().draw(x) and pca_plot().draw(x, y) is that:    
         - pca_plot().draw(x) will give a normal pca plot.    
         - pca_plot().draw(x, y) will coloring the points on the figure based on y. Set discrete_legend to True if y is continuous, False otherwise.    

        What we do here is:    
            1. standardize x    
            2. fit a PCA(n_pc) on x and storing the result in pd.DataFrame    
            3. Decide how to color the plots by various cases of y and discrete_legend.    

        Args:
            x (pd.DataFrame): feature    
            y (pd.Series, optional): label. Defaults to None.    
        """

        # calculate pca and store in pd.DataFrame
        pcs = PCA(self.n_pc).fit_transform((x - x.mean()) / (x.std() + 1e-4))
        pcs = pd.DataFrame(
            pcs,
            index=x.index,
            columns=["pc_" + str(i + 1) for i in range(self.n_pc)])

        if y is None:
            y_name = None
        else:
            # add y into the DataFrame for coloring the data points
            y_name = "y" if y.name is None else y.name
            pcs[y_name] = y

        if self.discrete_legend:
            # discrete legend
            plot = pairplot(data=pcs, hue=y_name)
        elif not y is None:
            # color bar
            cmap = color_palette('ch:', as_cmap=True)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize())
            sm.set_array([])
            plot = pairplot(data=pcs, hue=y_name)
            plot.legend.remove()

            cbar = plt.colorbar(sm, ax=plot.axes)
            cbar.set_label(y_name)
        else:
            # vanilla
            plot = pairplot(data=pcs)

        plot.figure.suptitle("{} {} Scatter plot".format(
            self.prefix, self.name),
                             y=1.01)


class pls_plot(basic_plot):
    """
    PLS-DA is a supervied method in dimension decomposition.    
    This function will plot the result of PLS-DA of given data.    

    Using pls_plot().make_figure(x, y) to generate a figure.

    Warning: PLS-DA is a limited tool in multi-class classification and unlinear regression problem.

    """

    def __init__(self,
                 is_classification: bool,
                 discrete_legend: bool = True,
                 prefix: str = "",
                 save_path: str = "./output/images/",
                 save_fig: bool = True,
                 show_fig: bool = True):
        """

        Args:
            is_classification (bool): If (x, y) is a classification task, then set is_classification to True.
            discrete_legend (bool, optional): To color the plot based on y in discrete hue or continuous color bar. If y is continuous, then you should set it to False. Defaults to True.
        """
        super().__init__(prefix=prefix,
                         save_path=save_path,
                         save_fig=save_fig,
                         show_fig=show_fig)
        self.discrete_legend = discrete_legend
        self.is_classification = is_classification
        self.name = "PLS plot"

    def reference(self) -> dict[str, str]:

        refer = super().reference()
        refer[
            self.name +
            " document"] = "https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html"

        return refer

    def draw(self, x: pd.DataFrame, y: pd.Series):
        """
        What we do here is:    
            1. If is_classification => one-hot encode the y. 
            2. Do PLS-DA
            3. Decide how to color the plots by various cases of y and discrete_legend.    

        Args:
            x (pd.DataFrame): features 
            y (pd.Series, optional): label.

        Raises:
            TypeError: _description_
        """
        # one hot encoder for classification
        if self.is_classification:
            OneHot_y = OneHotEncoder(sparse_output=False).fit_transform(
                y.to_numpy().reshape(-1, 1))
            # fit pls regression
            OneHot_y = (OneHot_y - OneHot_y.mean(
                axis=0, keepdims=True)) / OneHot_y.std(axis=0, keepdims=True)
            pls = PLSRegression(n_components=2).fit(x, OneHot_y)
        else:
            if y.dtype == "O":
                raise TypeError(
                    "the dtype of y can't be object while is_classification was setting to False, which means it is a regression task and y should be float or int."
                )
            # fit pls regression
            pls = PLSRegression(n_components=2).fit(x,
                                                    (y - y.mean()) / y.std())

        # project x
        plscs = pls.transform(x)
        plscs = pd.DataFrame(
            plscs,
            index=x.index,
            columns=[self.name + " componet 1", self.name + " componet 2"])

        if y is None:
            y_name = None
        else:
            # add y into the DataFrame for coloring the data points
            y_name = "y" if y.name is None else y.name
            plscs[y_name] = y

        if self.discrete_legend:
            # discrete legend
            plot = scatterplot(data=plscs,
                               x=self.name + " componet 1",
                               y=self.name + " componet 2",
                               hue=y_name)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        elif not y is None:
            # color bar
            cmap = color_palette('ch:', as_cmap=True)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize())
            plot = scatterplot(data=plscs,
                               x=self.name + " componet 1",
                               y=self.name + " componet 2",
                               hue=y_name,
                               hue_norm=sm.norm,
                               palette=cmap,
                               legend=False)
            cbar = plt.colorbar(sm, ax=plt.gca())
            cbar.set_label(y_name)
        else:
            # vanilla
            plot = scatterplot(data=plscs,
                               x=self.name + " componet 1",
                               y=self.name + " componet 2")

        plot.set_title("{} {} Scatter plot".format(self.prefix, self.name))


class umap_plot(basic_plot):
    """
    Umap is a unsupervissed method to reduce the number of dimension of given data. It's based on manifold learning and it has uncertainty.    

    Using umap_plot().make_figure(x) or umap_plot().make_figure(x, y) to generate a Umap plot.    
    
    Warning: Only the clustering tendency is reliable on the graph. 
    """

    def __init__(self,
                 discrete_legend=True,
                 prefix="",
                 save_path="./output/images/",
                 save_fig=True,
                 show_fig=True):
        """

        Args:
            discrete_legend (bool, optional): To color the plot based on y in discrete hue or continuous color bar. If y is continuous, then you should set it to False. Defaults to True.
        """
        super().__init__(prefix=prefix,
                         save_path=save_path,
                         save_fig=save_fig,
                         show_fig=show_fig)
        self.discrete_legend = discrete_legend
        self.name = "UMAP plot"

    def reference(self) -> dict[str, str]:

        refer = super().reference()
        refer[self.name +
              " document"] = "https://umap-learn.readthedocs.io/en/latest/"
        refer[
            self.name +
            " publication"] = "https://joss.theoj.org/papers/10.21105/joss.00861"

        return refer

    def draw(self, x: pd.DataFrame, y: pd.Series = None):
        """
        Using Umap do transform x into 2D plane. The difference between umap_plot().draw(x) and umap_plot().draw(x, y) is that:    
         - umap_plot().draw(x) will give a normal umap scatter plot.    
         - umap_plot().draw(x, y) will coloring the points on the figure based on y. Set discrete_legend to True if y is continuous, False otherwise.    

        What we do here is:    
            1. standardize x    
            2. fit a UMAP(n_neighbors=np.log2(n_sample) on x and storing the result in pd.DataFrame    
            3. Decide how to color the plots by various cases of y and discrete_legend.    

        Args:
            x (pd.DataFrame): feature    
            y (pd.Series, optional): label. Defaults to None.    
        """
        from umap import UMAP

        # fit a umap for x, you can change n_neighbors to any other feasible value.
        umapcs = UMAP(n_neighbors=round(np.log2(x.shape[0])),
                      n_components=2).fit_transform(
                          (x - x.mean()) / (x.std() + 1e-6))
        umapcs = pd.DataFrame(
            umapcs,
            index=x.index,
            columns=[self.name + " dimension 1", self.name + " dimension 2"])

        if y is None:
            y_name = None
        else:
            # add y into the DataFrame for coloring the data points
            y_name = "y" if y.name is None else y.name
            umapcs[y_name] = y

        if self.discrete_legend:
            # discrete legend
            plot = scatterplot(data=umapcs,
                               x=self.name + " dimension 1",
                               y=self.name + " dimension 2",
                               hue=y_name)
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        elif not y is None:
            # color bar
            cmap = color_palette('ch:', as_cmap=True)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize())
            plot = scatterplot(data=umapcs,
                               x=self.name + " dimension 1",
                               y=self.name + " dimension 2",
                               hue=y_name,
                               hue_norm=sm.norm,
                               palette=cmap,
                               legend=False)
            cbar = plt.colorbar(sm, ax=plt.gca())
            cbar.set_label(y_name)
        else:
            # vanilla
            plot = scatterplot(data=umapcs,
                               x=self.name + " dimension 1",
                               y=self.name + " dimension 2")

        plot.set_title("{} {} Scatter plot".format(self.prefix, self.name))


class corr_heatmap_plot(basic_plot):
    """
    plot correlation coefficients of given data in heatmap.    

    Using corr_heatmap_plot().make_figure(x) or corr_heatmap_plot().make_figure(x, y) to generate a figure.    
    """

    def __init__(self,
                 prefix="",
                 save_path="./output/images/",
                 save_fig=True,
                 show_fig=True):
        super().__init__(prefix=prefix,
                         save_path=save_path,
                         save_fig=save_fig,
                         show_fig=show_fig)
        self.name = "Correlation Heatmap plot"

    def draw(self, x: pd.DataFrame, y: pd.Series = None):
        """
        what we do here is:    
            1. add y into x as a new column if y is not None.    
            2. compute correlation between x's columns.
            3. drawint the result in heatmap.

        Args:
            x (pd.DataFrame): feature
            y (pd.Series, optional): Label. Defaults to None.
        """

        data = x.copy()
        if y is None:
            y_name = None
        else:
            # add y into the DataFrame for coloring the data points
            y_name = "y" if y.name is None else y.name
            data[y_name] = y

        plot = heatmap(data.corr(), vmin=-1, vmax=1, cmap='RdBu')

        plot.set_title("{} {}".format(self.prefix, self.name))


class confusion_matrix_plot(basic_plot):
    """
    plot confusion matrix of given ground true label and predictions.
    """

    def __init__(self,
                 prefix="",
                 save_path="./output/images/",
                 save_fig=True,
                 show_fig=True):
        """

        Todo:
            value normalize by y_true and crowding problem in multi-class classification.
        """
        super().__init__(prefix=prefix,
                         save_path=save_path,
                         save_fig=save_fig,
                         show_fig=show_fig)
        self.name = "Confusion Matrix"
        self.normalize = None

    def draw(self, y_true: pd.Series, y_pred: pd.Series):
        """
        plot the confusion matrix using y_true and y_pred.

        Args:
            y_true (pd.Series): Ground true
            y_pred (pd.Series): prediction from an estimator.
        """

        plot = metrics.ConfusionMatrixDisplay.from_predictions(
            y_true,
            y_pred,
            normalize=self.normalize,
            xticks_rotation="vertical")
        plot.ax_.set_title("{} {}".format(self.prefix, self.name))


class roc_plot(basic_plot):
    """
    Depends on how the number of class in y_true and, pos_label, this function will plot a roc curve or several curves of given data.    
    """

    def __init__(self,
                 pos_label: Union[str, int, float] = None,
                 prefix="",
                 save_path="./output/images/",
                 save_fig=True,
                 show_fig=True):
        """

        Args:
            pos_label (Union[str, int, float], optional): If not None, the result will be pos_label vs rest (ovr) roc curve. Defaults to None.
        """
        super().__init__(prefix=prefix,
                         save_path=save_path,
                         save_fig=save_fig,
                         show_fig=show_fig)
        self.name = "ROC Curve"
        self.pos_label = pos_label

    def draw(self, y_true: pd.Series, y_pred_prob: pd.DataFrame):
        """
        draw roc curve

        Args:
            y_true (pd.Series): Ground true
            y_pred_prob (pd.DataFrame): The probability that model predicted for each class. y_pred_prob should have shape (n_samples, n_class)
        """

        # ROC curve
        if len(y_true.value_counts()) <= 2:
            if self.pos_label is None:
                self.pos_label = y_true.iloc[0]
            # binary ROC curve
            fpr, tpr, threshold = metrics.roc_curve(
                y_true, y_pred_prob[self.pos_label], pos_label=self.pos_label)
            roc_auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
            plt.title(self.prefix + 'ROC curve')

        else:
            # one vs rest ROC curve
            for label in y_pred_prob.columns:
                label_prob = y_pred_prob[label]

                fpr, tpr, threshold = metrics.roc_curve(
                    y_true == label, label_prob)
                roc_auc = metrics.auc(fpr, tpr)

                plt.plot(fpr, tpr, label=str(label) + ' (AUC=%0.3f)' % roc_auc)
            plt.title(self.prefix + 'one vs rest ROC curves')

        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend(loc='lower right')


def data_overview(input_x: pd.DataFrame,
                  y: pd.Series,
                  is_classification: bool = True,
                  discrete_legend=True,
                  n_pc=4,
                  prefix="",
                  save_fig=True,
                  save_path="./output/images/",
                  show_fig=True):
    """
    Make a glance to data. Specifically it will:    
        1. make a pca plot.    
        2. make a pls plot.    
        3. make a Umap plot.   
        4. make a correlation heatmap.     

    Args:
        input_x (pd.DataFrame): the input feature (or the explanatory variables). 
        y (pd.Series): the target variable (or the response variable).
        is_classification (bool, optional): whether the task to fit y is a classification task or a regression task. Defaults to True.
        discrete_legend (bool, optional): whether to use a discrete legend, otherwise a color bar will be used. If is_classification is True, then discrete_legend will forced to be Ture. Defaults to True.
    """

    x = input_x.copy()
    if is_classification:
        discrete_legend = True

    # PCA
    pca_plot(n_pc=n_pc,
             discrete_legend=discrete_legend,
             prefix=prefix,
             save_path=save_path,
             save_fig=save_fig,
             show_fig=show_fig).make_figure(x, y)

    # PLS
    pls_plot(is_classification=is_classification,
             discrete_legend=discrete_legend,
             prefix=prefix,
             save_path=save_path,
             save_fig=save_fig,
             show_fig=show_fig).make_figure(x, y)

    # UMAP
    umap_plot(discrete_legend=discrete_legend,
              prefix=prefix,
              save_path=save_path,
              save_fig=save_fig,
              show_fig=show_fig).make_figure(x, y)

    # Correlation heatmap
    if y.dtype == "O":
        corr_heatmap_plot(prefix=prefix,
                          save_path=save_path,
                          save_fig=save_fig,
                          show_fig=show_fig).make_figure(x)
    else:
        corr_heatmap_plot(prefix=prefix,
                          save_path=save_path,
                          save_fig=save_fig,
                          show_fig=show_fig).make_figure(x, y)


def classification_summary(y_true,
                           y_pred_prob,
                           target_label=None,
                           prefix="",
                           save_path="./output/images/",
                           save_fig=True,
                           show_fig=True):
    """
    Give a classification summary, including:    
        1. recall, precision, f1 and accuracy.    
        2. confusion matrix.    
        3. ROC curve.    

    Args:
        y_true (pandas.Series or a 1D array): The label.
        y_pred_prob (pandas.Series or a 1D array): float in [0, 1]. prediction from model.
        target_label ( member of label, optional): The target class from y_true to compute sensitivity and specificity. Defaults to None.
    Todo:
        1. support to multi-class classification(on going)
        2. the label matching between y_true and y_pred
        3. support to regression
    """
    y_pred = y_pred_prob.idxmax(axis=1)
    #print("\n", prefix)
    #print(metrics.classification_report(y_true, y_pred))

    report = metrics.classification_report(y_true, y_pred, output_dict=True)
    acc = report.pop("accuracy")
    report = pd.DataFrame(report).T
    report.loc["accuracy"] = [" ", " ", acc, y_true.count()]

    if not target_label is None:
        # binary classification
        prf1 = metrics.classification_report(y_true == target_label,
                                             y_pred == target_label,
                                             output_dict=True)

        report.loc["sensitivity"] = [
            " ", " ", prf1["True"]["recall"], prf1["True"]["support"]
        ]
        report.loc["specificity"] = [
            " ", " ", prf1["False"]["recall"], prf1["True"]["support"]
        ]

        #print("sensitivity: {:.3f}".format(sensitivity))
        #print("specificity: {:.3f}".format(specificity))

    # insert an empty row to split
    tmp = report.loc[y_pred_prob.columns.astype(str)]
    tmp.loc["   "] = [" ", " ", " ", " "]
    report = pd.concat([tmp, report.drop(y_pred_prob.columns.astype(str))],
                       axis=0)

    print(report)
    if save_fig:
        report.to_csv(save_path + "{} scores.csv".format(prefix))

    # confusion matrix
    confusion_matrix_plot(prefix=prefix,
                          save_path=save_path,
                          show_fig=show_fig,
                          save_fig=save_fig).make_figure(y_true, y_pred)

    # roc cuve
    roc_plot(pos_label=target_label,
             prefix=prefix,
             save_path=save_path,
             show_fig=show_fig,
             save_fig=save_fig).make_figure(y_true, y_pred_prob)


def regression_summary(y_true: pd.Series,
                       y_pred: pd.Series,
                       x: pd.DataFrame = None,
                       prefix="",
                       save_path="./output/images/",
                       save_fig=True,
                       show_fig=True):
    """
    1. compute rmse, r square and mape if y is all positive.    
    2. feature pca residual plot if x is not None.

    Args:
        y_true (pd.Series): Ground true
        y_pred (pd.Series): The estimates from model.
        x (pd.DataFrame, optional): features. If not None, it will be used to plot a feature pca residual plot. Defaults to None.
    """

    scores = pd.Series(
        {
            "rmse": metrics.root_mean_squared_error(y_true, y_pred),
            "r2": metrics.r2_score(y_true, y_pred),
            "mae": metrics.mean_absolute_error(y_true, y_pred),
        },
        name="performance")
    if (y_true > 0).all():
        scores["mape"] = metrics.mean_absolute_percentage_error(y_true, y_pred)
    scores["support"] = len(y_true)

    print("\n", prefix, " performance:")
    print(scores)
    if save_fig:
        scores.to_csv(save_path + "{} scores.csv".format(prefix))

    residual = y_true - y_pred
    if not x is None:
        pca_plot(n_pc=2,
                 discrete_legend=False,
                 prefix=prefix + " Residual",
                 save_path=save_path,
                 save_fig=save_fig,
                 show_fig=show_fig).make_figure(x, residual)
