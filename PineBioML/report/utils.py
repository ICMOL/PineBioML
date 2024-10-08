import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from seaborn import scatterplot, heatmap, pairplot

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import OneHotEncoder
import sklearn.metrics as metrics

from umap import UMAP


def plot_pca(x,
             y,
             n_pc=4,
             prefix="",
             save_path="./output/images/",
             save_fig=True,
             show_fig=True):

    print("start PCA")
    pcs = PCA(n_pc).fit_transform((x - x.mean()) / x.std())
    pcs = pd.DataFrame(pcs,
                       index=x.index,
                       columns=["pc_" + str(i + 1) for i in range(4)])

    y_name = "y" if y.name is None else y.name
    pcs[y_name] = y

    plot = pairplot(pcs, hue=y_name)
    plot.figure.suptitle(prefix + " PCA Scatter plot", y=1.01)

    if save_fig:
        plt.savefig(save_path + prefix + " PCA Scatter plot")
    if show_fig:
        plt.show()
    else:
        plt.clf()


def plot_pls(x,
             y,
             is_classification=True,
             prefix="",
             save_path="./output/images/",
             save_fig=True,
             show_fig=True):
    print("start PLS")

    # one hot encoder for classification
    if is_classification:
        OneHot_y = OneHotEncoder(sparse_output=False).fit_transform(
            y.to_numpy().reshape(-1, 1))

    # fit pls regression
    pls = PLSRegression(n_components=2).fit(x, OneHot_y)
    # project x
    plscs = pls.transform(x)
    plscs = pd.DataFrame(plscs,
                         index=x.index,
                         columns=["pls componet 1", "pls componet 2"])

    # adding y into x to draw the plot
    y_name = "y" if y.name is None else y.name
    plscs[y_name] = y

    # seaborn scatter plot
    fig = scatterplot(data=plscs,
                      x="pls componet 1",
                      y="pls componet 2",
                      hue=y_name).set_title("PLS")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # to save and show
    if save_fig:
        plt.savefig(save_path + prefix + " PLS")
    if show_fig:
        plt.show()
    else:
        plt.clf()


def plot_umap(x,
              y,
              prefix="",
              save_path="./output/images/",
              save_fig=True,
              show_fig=True):
    print("start UMAP")

    # fit a umap for x, you can change n_neighbors to any other feasible value.
    umapcs = UMAP(n_neighbors=round(np.log2(x.shape[0])),
                  n_components=2).fit_transform(x)
    umapcs = pd.DataFrame(umapcs,
                          index=x.index,
                          columns=["umap dimension 1", "umap dimension 2"])

    # adding y into x to draw the plot
    y_name = "y" if y.name is None else y.name
    umapcs[y_name] = y

    # seaborn scatter plot
    fig = scatterplot(data=umapcs,
                      x="umap dimension 1",
                      y="umap dimension 2",
                      hue=y_name).set_title("umap")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # to save and show
    if save_fig:
        plt.savefig(save_path + prefix + " umap")
    if show_fig:
        plt.show()
    else:
        plt.clf()


def plot_corr_heatmap(x,
                      y=None,
                      prefix="",
                      save_path="./output/images/",
                      save_fig=True,
                      show_fig=True):
    # corr heat map
    fig = heatmap(x.corr(), vmin=-1, vmax=1,
                  cmap='RdBu').set_title(prefix + " Variable Correlation")
    # to save and show
    if save_fig:
        plt.savefig(save_path + prefix + " Variable Correlation")
    if show_fig:
        plt.show()
    else:
        plt.clf()


def data_overview(input_x,
                  y,
                  is_classification=True,
                  prefix="",
                  save_fig=True,
                  save_path="./output/images/",
                  show_fig=True):
    """
    Give a data overview, including:    
        1. raw data boxplot, pairplot.        
        2. pca pairt plot.     
        3. corelation heatmap.    

    Args:
            input_x (pandas.DataFrame): the data
            y (pandas.Series or a 1D array): label
            label_name (str): the name of y
            prefix (str): the name of data
    """

    x = input_x.copy()

    # PCA
    plot_pca(x,
             y,
             n_pc=4,
             prefix=prefix,
             save_path=save_path,
             save_fig=save_fig,
             show_fig=show_fig)

    #
    plot_pls(x,
             y,
             is_classification=is_classification,
             prefix=prefix,
             save_path=save_path,
             save_fig=save_fig,
             show_fig=show_fig)

    # Correlation heatmap
    plot_corr_heatmap(x,
                      y,
                      prefix=prefix,
                      save_path=save_path,
                      save_fig=save_fig,
                      show_fig=show_fig)

    # UMAP
    plot_umap(x,
              y,
              prefix=prefix,
              save_path=save_path,
              save_fig=save_fig,
              show_fig=show_fig)


def classification_scores(fitted_model, x, y_true, prefix=""):
    """
    compute classification metrics, including acc, f1, recall, specificity, sensitivity, mcc, auc

    Args:
        fitted_model (model): fitted model
        x (pd.DataFrame): features
        y_true (pd.Series): ground true lebel
        prefix (str): score prefix

    Returns:
        dict: python dict for scores
    """
    y_pred_prob = fitted_model.predict_proba(x)
    y_pred = y_pred_prob.idxmax(axis=1)

    if len(y_true.value_counts()) <= 2:
        # binary classification
        confusion_scores = metrics.classification_report(y_true,
                                                         y_pred,
                                                         output_dict=True)

        if len(confusion_scores) == 5:
            if "1" in confusion_scores:
                result = confusion_scores["1"]
                result["sensitivity"] = confusion_scores["1"]["recall"]
                result["specificity"] = confusion_scores["0"]["recall"]
                #sensitivity = confusion_scores["1"]["recall"]
                #specificity = confusion_scores["0"]["recall"]

            if "1.0" in confusion_scores:
                result = confusion_scores["1.0"]
                result["sensitivity"] = confusion_scores["1.0"]["recall"]
                result["specificity"] = confusion_scores["0.0"]["recall"]
                #sensitivity = confusion_scores["1.0"]["recall"]
                #specificity = confusion_scores["0.0"]["recall"]

        result["mcc"] = metrics.matthews_corrcoef(y_true, y_pred)
        result["auc"] = metrics.roc_auc_score(y_true, y_pred_prob.iloc[:, 1])
        result["support"] = len(y_true)

        prefix_result = {}
        for score in result:
            prefix_result[prefix + score] = result[score]
    else:
        # multi-class classification
        result = {}
        result["cross_entropy"] = metrics.log_loss(y_true, y_pred_prob)
        result["balanced_accuracy"] = metrics.balanced_accuracy_score(
            y_true, y_pred)
        result["cohen_kappa"] = metrics.cohen_kappa_score(y_true, y_pred)
        result["support"] = len(y_true)

        prefix_result = {}
        for score in result:
            prefix_result[prefix + score] = result[score]

    return prefix_result


def classification_summary(y_true,
                           y_pred_prob,
                           prefix="",
                           model=None,
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

    if len(y_true.value_counts()) <= 2:
        # binary classification
        print("\n", prefix)
        print(metrics.classification_report(y_true, y_pred))

        confusion_scores = metrics.classification_report(y_true,
                                                         y_pred,
                                                         output_dict=True)
        if len(confusion_scores) == 5:
            if "1" in confusion_scores:
                sensitivity = confusion_scores["1"]["recall"]
                specificity = confusion_scores["0"]["recall"]
            if "1.0" in confusion_scores:
                sensitivity = confusion_scores["1.0"]["recall"]
                specificity = confusion_scores["0.0"]["recall"]
            print("sensitivity: {:.3f}".format(sensitivity))
            print("specificity: {:.3f}".format(specificity))

        # confusion matrix
        metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
        if save_fig:
            plt.savefig(save_path + prefix + " confussion_matrix")
        if show_fig:
            plt.show()
        else:
            plt.clf()

        # ROC curve
        fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred_prob.iloc[:, 1])
        roc_auc = metrics.auc(fpr, tpr)

        plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title(prefix + ' ROC curve')
        plt.legend(loc='lower right')

        plt.tight_layout()
        if save_fig:
            plt.savefig(save_path + prefix + " ROC curve")
        if show_fig:
            plt.show()
        else:
            plt.clf()

    else:
        # multi-class classification
        print("\n", prefix)
        print(metrics.classification_report(y_true, y_pred))

        confusion_scores = metrics.classification_report(y_true,
                                                         y_pred,
                                                         output_dict=True)
        # confusion matrix
        metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
        plt.xticks(rotation=90)
        plt.tight_layout()
        if save_fig:
            plt.savefig(save_path + prefix + " confussion_matrix")
        if show_fig:
            plt.show()
        else:
            plt.clf()

        # one vs rest ROC curve
        for label in y_pred_prob.columns:
            label_prob = y_pred_prob[label]

            fpr, tpr, threshold = metrics.roc_curve(y_true == label,
                                                    label_prob)
            roc_auc = metrics.auc(fpr, tpr)

            plt.plot(fpr,
                     tpr,
                     label=str(label) + ' vs rest (AUC=%0.3f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.title(prefix + 'one vs rest ROC curves')
        plt.legend(loc='lower right')

        plt.tight_layout()

        if save_fig:
            plt.savefig(save_path + prefix + " ROC curve")
        if show_fig:
            plt.show()
        else:
            plt.clf()
