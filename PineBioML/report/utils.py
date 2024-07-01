import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
import sklearn.metrics as metrics


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

    n_feature = x.shape[1]
    max_display_number = 12
    n_pc = min(n_feature, max_display_number)

    # PCA
    pcs = PCA(n_pc).fit_transform((x - x.mean()) / x.std())

    # turn into pandas dataframe format
    pcs = pd.DataFrame(pcs,
                       index=x.index,
                       columns=["pc_" + str(i) for i in range(n_pc)])

    x[label_name] = y
    pcs[label_name] = y

    if n_feature > max_display_number:
        print("x has too many features(>", max_display_number,
              "), pass boxplot")
    else:
        # variable box plot
        fig = sns.boxplot(x)
        fig.set_title(title + " Variable Boxplot")
        for label in fig.get_xticklabels(which="major"):
            label.set(rotation=45, horizontalalignment='right')
        fig.set_ylabel("values")
        plt.tight_layout()
        plt.show()

    if n_feature > max_display_number * 2:
        print("x has too many features(>", max_display_number * 2,
              "), pass correlation heatmap")
    else:
        # correlation heatmap
        fig = sns.heatmap(x.corr(), vmin=-1, vmax=1,
                          cmap='BrBG').set_title(title +
                                                 " Variable Correlation")
        plt.tight_layout()
        plt.show()

    if n_feature > max_display_number:
        print("x has too many features(>", max_display_number,
              "), pass pairplot")
    else:
        # pair plot
        print(title + " Paired Scatter plot")
        fig = sns.pairplot(x, hue=label_name)
        plt.show()

    # pca pair plot
    print(title + " PCA Scatter plot")
    # plot up to 12 principle complenent
    fig = sns.pairplot(pcs, hue=label_name)
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
