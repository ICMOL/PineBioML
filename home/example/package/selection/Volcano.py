import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t
from .base import selection as base_selection

class Volcano_selection(base_selection):
    def __init__(self, plotting = False, strategy = "fold", p_threshold = 0.05, fc_threshold = 2):
        super().__init__()
        self.plotting = plotting
        self.strategy = strategy
        self.fc_threshold = fc_threshold
        self.p_threshold = 0.05

    def scoring(slef, x, y):
        positive = y == 1
        negative = y == 0

        # fold change
        #log_fold = np.log2(x[positive].mean(axis= 0)/ x[negative].mean(axis= 0))
        log_fold = x[positive].mean(axis= 0)- x[negative].mean(axis= 0)

        # Welch t test:
        #     normal assumption
        #     diffirent sample size
        #     diffirent varience
        #     unpaired
        n_positive = x[positive].shape[0]
        n_negative= x[negative].shape[0]
        diff = x[positive].mean(axis= 0)- x[negative].mean(axis= 0)

        s_positive = ((x[positive] - x[positive].mean(axis = 0))**2).sum(axis = 0)/(n_positive -1)
        s_negative = ((x[negative] - x[negative].mean(axis = 0))**2).sum(axis = 0)/(n_negative -1)
        st = np.sqrt(s_positive/n_positive + s_negative/n_negative)

        t_statistic = np.abs(diff/st)
        df = (s_positive/n_positive + s_negative/n_negative)**2 /(
            (s_positive/n_positive)**2 / (n_positive-1) +
            (s_negative/n_negative)**2 / (n_negative-1)
        )

        # 2 side testing
        print("t statistic: ", t_statistic.min(), t_statistic.mean(), t_statistic.max())
        print("degree of freedom: ", df.min(), df.mean(), df.max())

        p_value = t.cdf(x= -t_statistic, df=df)*2
        log_p = -np.log10(p_value)

        return pd.DataFrame({"log_p_value": log_p, "log_fold_change": log_fold}, index = log_fold.index)

    def choose(self, score, k):
        log_fold = score["log_fold_change"]
        log_p = score["log_p_value"]
        # choose fold change > 2 and p value < 0.05 in log scale
        significant = np.logical_and(np.abs(log_fold)>= np.log2(self.fc_threshold),  log_p> -np.log10(self.p_threshold))
        
        # choose top k logged p-value 
        if self.strategy == "fold":
            tmp = np.abs(log_fold)
            selected = np.zeros(len(tmp[significant]), dtype = np.bool_)
            selected[np.argsort(tmp[significant])[-k:]] = True
        elif self.strategy == "p":
            selected = np.zeros(len(log_p[significant]), dtype = np.bool_)
            selected[np.argsort(log_p[significant])[:k]] = True
        else:
            raise "select_by must be one of {fold} or {p}"

        if self.plotting :
            # silent
            plt.scatter(x = log_fold[~significant], y = log_p[~significant], s = 0.5, color = 'gray')
            # not selected
            plt.scatter(x = log_fold[significant][~selected], y = log_p[significant][~selected], s = 2)
            # selected
            plt.scatter(x = log_fold[significant][selected], y = log_p[significant][selected], s = 2)

            plt.title("Welch t-test volcano")
            plt.xlabel("log_2 fold")
            plt.axvline(1, linestyle="dotted", color = "gray")
            plt.axvline(-1, linestyle="dotted", color = "gray")
            plt.axhline(-np.log10(0.05), linestyle="dotted", color = "gray")
            plt.ylabel("log_10 p")
            plt.show()


        # volcano plot importance
        return pd.Series(log_p[significant][selected], index = log_fold.index[significant][selected], name = "Volcano Plot_"+ self.strategy).sort_values()


