import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from seaborn import scatterplot, heatmap, pairplot

from sklearn.decomposition import PCA
import sklearn.metrics as metrics
from sklearn.cross_decomposition import PLSRegression
from umap import UMAP


class task_card:
    """ 
    task information
    """

    def __init__(self, **parms):
        self.parms = parms


class data_card:
    """ 
    task information
    """

    def __init__(self, data):
        pass


def data_overview(input_x, y, label_name="y", title=""):
    """
    Give a data overview, including:
        1. raw data boxplot, pairplot.    
        2. pca pairt plot.    
        3. corelation heatmap.

    Args:
            input_x (pandas.DataFrame): the data
            y (pandas.Series or a 1D array): label
            label_name (str): the name of y
            title (str): the name of data
    """

    x = input_x.copy()

    # PCA
    print("start PCA")
    pcs = PCA(4).fit_transform((x - x.mean()) / x.std())
    # turn into pandas dataframe format
    pcs = pd.DataFrame(pcs,
                       index=x.index,
                       columns=["pc_" + str(i + 1) for i in range(4)])

    # PLS
    print("start PLS")
    pls = PLSRegression().fit(x, y)
    plscs = pls.transform(x)
    plscs = pd.DataFrame(plscs,
                         index=x.index,
                         columns=["pls componet 1", "pls componet 2"])

    # UMAP
    print("start UMAP")
    umapcs = UMAP(n_neighbors=round(np.log2(x.shape[0])),
                  n_components=2).fit_transform(x)
    umapcs = pd.DataFrame(umapcs,
                          index=x.index,
                          columns=["umap dimension 1", "umap dimension 2"])

    x[label_name] = y
    pcs[label_name] = y
    plscs[label_name] = y
    umapcs[label_name] = y

    # plotting
    ### correlation heatmap
    fig = heatmap(x.corr(), vmin=-1, vmax=1,
                  cmap='RdBu').set_title(title + " Variable Correlation")
    plt.tight_layout()
    plt.show()

    ### pca pair plot
    print(title + " PCA Scatter plot")
    # plot up to 12 principle complenent
    fig = pairplot(pcs, hue=label_name)
    #sns.scatterplot(pcs[["pc_1", "pc_2", label_name]], hue=label_name)
    plt.show()

    ### Umap
    fig = scatterplot(data=umapcs,
                      x="umap dimension 1",
                      y="umap dimension 2",
                      hue=label_name)
    fig.set_title("umap")
    plt.show()

    ### PLS
    fig = scatterplot(data=plscs,
                      x="pls componet 1",
                      y="pls componet 2",
                      hue=label_name)
    fig.set_title("PLS")
    plt.show()


def classification_summary(y_true, y_pred_prob, title=""):
    """
    Give a classification summary, including:
        1. recall, precision, f1 and accuracy.    
        2. confusion matrix.    
        3. ROC curve.

    Args:
            y_true (pandas.Series or a 1D array): Bool, the label
            y_pred_prob (pandas.Series or a 1D array): float in [0, 1]. prediction from model
    """
    y_pred = np.round(y_pred_prob)

    print("\n", title)
    print(metrics.classification_report(y_true, y_pred))

    # confusion matrix
    metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.tight_layout()
    plt.show()

    # ROC curve
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred_prob)
    roc_auc = metrics.auc(fpr, tpr)

    plt.plot(fpr, tpr, 'b', label='AUC = %0.3f' % roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title(title + ' ROC curve')
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()
    """
    fig, ax = plt.subplots(1, 2)
    # confusion matrix
    metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax = ax[0])
    
    # ROC curve
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred_prob)
    roc_auc = metrics.auc(fpr, tpr)
    
    ax[1].plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    ax[1].plot([0, 1], [0, 1],'r--')
    ax[1].set_xlim([0, 1])
    ax[1].set_ylim([0, 1])
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_title(title+' ROC curve')
    ax[1].legend(loc = 'lower right')
    
    plt.tight_layout()
    plt.show()
    """
