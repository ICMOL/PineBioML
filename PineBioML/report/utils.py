from abc import ABC, abstractmethod

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from seaborn import scatterplot, heatmap, pairplot, color_palette

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import OneHotEncoder
import sklearn.metrics as metrics

from umap import UMAP


class basic_plot(ABC):
    """
    the base class of plots.
    """

    def __init__(self, prefix, save_path, save_fig, show_fig):
        """

        Args:
            prefix (str): the describe or title for the plot. the prefix will be added into the title of plot and saving name.
            save_path (str): the path to export the figure.
            save_fig (_type_): whether to export the figure or not.
            show_fig (_type_): whether to show the figure or not.
        """
        self.prefix = prefix
        self.save_path = save_path
        self.save_fig = save_fig
        self.show_fig = show_fig

    @abstractmethod
    def save_name(self):
        pass

    @abstractmethod
    def draw(self, x, y):
        pass

    def make_figure(self, x, y=None):
        self.draw(x, y)

        if self.save_fig:
            plt.savefig(self.save_path + self.save_name())
        if self.show_fig:
            plt.show()
        else:
            plt.clf()


class pca_plot(basic_plot):

    def __init__(self,
                 n_pc=4,
                 discrete_legend=True,
                 prefix="",
                 save_path="./output/images/",
                 save_fig=True,
                 show_fig=True):
        super().__init__(prefix=prefix,
                         save_path=save_path,
                         save_fig=save_fig,
                         show_fig=show_fig)
        self.n_pc = n_pc
        self.discrete_legend = discrete_legend
        self.name = "PCA"

    def save_name(self):
        return "{} {} plot".format(self.prefix, self.name)

    def draw(self, x, y=None):
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

    def __init__(self,
                 is_classification,
                 discrete_legend=True,
                 prefix="",
                 save_path="./output/images/",
                 save_fig=True,
                 show_fig=True):
        super().__init__(prefix=prefix,
                         save_path=save_path,
                         save_fig=save_fig,
                         show_fig=show_fig)
        self.discrete_legend = discrete_legend
        self.is_classification = is_classification
        self.name = "PLS"

    def save_name(self):
        return "{} {} plot".format(self.prefix, self.name)

    def draw(self, x, y=None):
        # one hot encoder for classification
        if self.is_classification:
            OneHot_y = OneHotEncoder(sparse_output=False).fit_transform(
                y.to_numpy().reshape(-1, 1))
            # fit pls regression
            pls = PLSRegression(n_components=2).fit(x, OneHot_y)
        else:
            if y.dtype == "O":
                raise TypeError(
                    "the dtype of y can't be object while is_classification was setting to False, which means it is a regression task and y should be float or int."
                )
            # fit pls regression
            pls = PLSRegression(n_components=2).fit(x, y)

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

    def __init__(self,
                 discrete_legend=True,
                 prefix="",
                 save_path="./output/images/",
                 save_fig=True,
                 show_fig=True):
        super().__init__(prefix=prefix,
                         save_path=save_path,
                         save_fig=save_fig,
                         show_fig=show_fig)
        self.discrete_legend = discrete_legend
        self.name = "UMAP"

    def save_name(self):
        return "{} {} plot".format(self.prefix, self.name)

    def draw(self, x, y=None):
        # fit a umap for x, you can change n_neighbors to any other feasible value.
        umapcs = UMAP(n_neighbors=round(np.log2(x.shape[0])),
                      n_components=2).fit_transform(x)
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

    def __init__(self,
                 prefix="",
                 save_path="./output/images/",
                 save_fig=True,
                 show_fig=True):
        super().__init__(prefix=prefix,
                         save_path=save_path,
                         save_fig=save_fig,
                         show_fig=show_fig)
        self.name = "Correlation Heatmap"

    def save_name(self):
        return "{} {} plot".format(self.prefix, self.name)

    def draw(self, x, y=None):
        data = x.copy()
        if y is None:
            y_name = None
        else:
            # add y into the DataFrame for coloring the data points
            y_name = "y" if y.name is None else y.name
            data[y_name] = y

        plot = heatmap(data.corr(), vmin=-1, vmax=1, cmap='RdBu')

        plot.set_title("{} {}".format(self.prefix, self.name))


class confustion_matrix_plot(basic_plot):

    def __init__(self,
                 prefix="",
                 save_path="./output/images/",
                 save_fig=True,
                 show_fig=True):
        super().__init__(prefix=prefix,
                         save_path=save_path,
                         save_fig=save_fig,
                         show_fig=show_fig)
        self.name = "Confusion Matrix"

    def save_name(self):
        return "{} {}".format(self.prefix, self.name)

    def draw(self, y_true, y_pred):
        metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
        plt.xticks(rotation=90)


class roc_plot(basic_plot):

    def __init__(self,
                 prefix="",
                 save_path="./output/images/",
                 save_fig=True,
                 show_fig=True):
        super().__init__(prefix=prefix,
                         save_path=save_path,
                         save_fig=save_fig,
                         show_fig=show_fig)
        self.name = "ROC Curve"

    def save_name(self):
        return "{} {}".format(self.prefix, self.name)

    def draw(self, y_true, y_pred_prob):
        # ROC curve
        if len(y_true.value_counts()) <= 2:
            # binary ROC curve
            fpr, tpr, threshold = metrics.roc_curve(y_true,
                                                    y_pred_prob.iloc[:, 1])
            roc_auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)

        else:
            # one vs rest ROC curve
            for label in y_pred_prob.columns:
                label_prob = y_pred_prob[label]

                fpr, tpr, threshold = metrics.roc_curve(
                    y_true == label, label_prob)
                roc_auc = metrics.auc(fpr, tpr)

                plt.plot(fpr, tpr, label=str(label) + ' (AUC=%0.3f)' % roc_auc)

        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title(self.prefix + 'one vs rest ROC curves')
        plt.legend(loc='lower right')


def data_overview(input_x,
                  y,
                  is_classification=True,
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
        prefix (str, optional): the name or the description to the task. It will be add into the title and the export figure name. Defaults to "".
        save_fig (bool, optional): To export the figure or not. Defaults to True.
        save_path (str, optional): The export path of the figure. Defaults to "./output/images/".
        show_fig (bool, optional): To show the figure before export or not. Defaults to True.
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
            y_true (pandas.Series or a 1D array): Bool, the label
            y_pred_prob (pandas.Series or a 1D array): float in [0, 1]. prediction from model
    Todo:
        1. support to multi-class classification(on going)
        2. the label matching between y_true and y_pred
        3. support to regression
    """
    y_pred = y_pred_prob.idxmax(axis=1)
    print("\n", prefix)
    print(metrics.classification_report(y_true, y_pred))

    if not target_label is None:
        # binary classification
        confusion_scores = metrics.classification_report(
            y_true == target_label, y_pred == target_label, output_dict=True)
        sensitivity = confusion_scores[True]["recall"]
        specificity = confusion_scores[False]["recall"]

        print("sensitivity: {:.3f}".format(sensitivity))
        print("specificity: {:.3f}".format(specificity))

    # confusion matrix
    confustion_matrix_plot(prefix=prefix,
                           save_path=save_path,
                           show_fig=show_fig,
                           save_fig=save_fig).make_figure(y_true, y_pred)

    # roc cuve
    roc_plot(prefix=prefix,
             save_path=save_path,
             show_fig=show_fig,
             save_fig=save_fig).make_figure(y_true, y_pred_prob)


def regression_summary(x,
                       y_true,
                       y_pred,
                       prefix="",
                       save_path="./output/images/",
                       save_fig=True,
                       show_fig=True):
    residual = y_true - y_pred

    print("\n", prefix, " performance:")
    print("    r2     : {:.3f}".format(
        metrics.root_mean_squared_error(y_true, y_pred)))
    print("    rmse   : {:.3f}".format(metrics.r2_score(y_true, y_pred)))
    print("    mae    : {:.3f}".format(
        metrics.mean_absolute_error(y_true, y_pred)))
    if (y_true > 0).all():
        print("    mape   : {:.3f}".format(
            metrics.mean_absolute_percentage_error(y_true, y_pred)))
    print("    support: {:.3f}".format(len(y_true)))

    pca_plot(n_pc=2,
             discrete_legend=False,
             prefix=prefix,
             save_path=save_path,
             save_fig=save_fig,
             show_fig=show_fig).make_figure(x, residual)
