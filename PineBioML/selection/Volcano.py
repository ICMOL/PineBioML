import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t
from . import SelectionPipeline


class Volcano_selection(SelectionPipeline):
    """
    volcano plot.

    """

    def __init__(self,
                 k,
                 z_importance_threshold=1.,
                 strategy="fold",
                 p_threshold=0.05,
                 fc_threshold=2,
                 log_domain=False,
                 absolute=True,
                 target_label=1):
        """

        Args:
            strategy (str, optional): Choosing strategy. One of {"p" or "fold"} Defaults to "fold".
            p_threshold (float, optional): p-value threshold. Only feature has p-value higher than threshold will be considered. Defaults to 0.05.
            fc_threshold (int, optional): fold change threshold. Only feature has fold change higher than threshold will be considered. Defaults to 2.
            log_domain (bool, optional): Whether input data is in log_domain. Defaults to False.
            absolute (bool, optional): If true, then take absolute value on score while strategy == "p". Defaults to True.
            target_label : the target label.
        """
        super().__init__(k=k, z_importance_threshold=z_importance_threshold)
        self.strategy = strategy
        self.fc_threshold = fc_threshold
        self.p_threshold = p_threshold
        self.log_domain = log_domain
        self.absolute = absolute
        self.name = "Volcano Plot_" + self.strategy
        self.missing_value = 0
        self.target_label = target_label

    def Scoring(self, x, y):
        """
        Compute the fold change and p-value on each feature.

        Args:
            x (pandas.DataFrame or a 2D array): The data to extract information.
            y (pandas.Series or a 1D array): The target label for methods. Defaults to None.

        Returns:
           pandas.DataFrame: A dataframe records p-value and fold change.
        """
        positive = y == self.target_label
        negative = np.logical_not(positive)

        x = x.replace(0, np.nan)

        # fold change
        if not self.log_domain:
            log_fold = np.log2(x[positive].mean(axis=0) /
                               x[negative].mean(axis=0))
        else:
            log_fold = x[positive].mean(axis=0) - x[negative].mean(axis=0)

        # Welch t test:
        #     normal assumption
        #     diffirent sample size
        #     diffirent varience
        #     unpaired
        n_positive = x[positive].shape[0]
        n_negative = x[negative].shape[0]
        diff = x[positive].mean(axis=0) - x[negative].mean(axis=0)

        s_positive = ((x[positive] - x[positive].mean(axis=0))**
                      2).sum(axis=0) / (n_positive - 1)
        s_negative = ((x[negative] - x[negative].mean(axis=0))**
                      2).sum(axis=0) / (n_negative - 1)
        st = np.sqrt(s_positive / n_positive + s_negative / n_negative)

        t_statistic = np.abs(diff / st)
        df = (s_positive / n_positive + s_negative / n_negative)**2 / (
            (s_positive / n_positive)**2 / (n_positive - 1) +
            (s_negative / n_negative)**2 / (n_negative - 1))

        # 2 side testing
        #print("t statistic: ", t_statistic.min(), t_statistic.mean(), t_statistic.max())
        #print("degree of freedom: ", df.min(), df.mean(), df.max())

        p_value = t.cdf(x=-t_statistic, df=df) * 2
        log_p = -np.log10(p_value)

        scores = pd.DataFrame(
            {
                "log_p_value": log_p,
                "log_fold_change": log_fold
            },
            index=log_fold.index)
        return scores

    def Select(self, scores):
        """
        Choosing the features which has score higher than threshold in assigned strategy.

        If strategy == "fold": sort in fold change and return p-value

        If strategy == "p": sort in p-value and return fold change

        Args:
            scores (pandas.DataFrame): A dataframe records p-value and fold change.
            k (int): Number of features to select.

        Returns:
            pandas.Series: The score for k selected features in assigned strategy.
        """
        log_fold = scores["log_fold_change"]
        log_p = scores["log_p_value"]
        # choose fold change > 2 and p value < 0.05 in log scale
        significant = np.logical_and(
            np.abs(log_fold) >= np.log2(self.fc_threshold), log_p
            > -np.log10(self.p_threshold))
        self.significant = significant

        # choose top k logged p-value
        if self.strategy == "fold":
            selected = np.abs(log_fold).loc[significant].sort_values().tail(
                self.k)
            selected_score = pd.Series(log_p.loc[selected.index],
                                       index=selected.index,
                                       name=self.name)
        elif self.strategy == "p":
            selected = log_p.loc[significant].sort_values().tail(self.k)
            if self.absolute:
                selected_score = pd.Series(
                    np.abs(log_fold.loc[selected.index]),
                    index=selected.index,
                    name=self.name).sort_values(ascending=False)
            else:
                selected_score = pd.Series(
                    log_fold.loc[selected.index],
                    index=selected.index,
                    name=self.name).sort_values(ascending=False)
        else:
            raise "select_by must be one of {fold} or {p}"

        return selected_score

    def plotting(self,
                 external=False,
                 external_score=None,
                 title="Welch t-test volcano",
                 show=True,
                 saving=False,
                 save_path="./output/"):
        """
        Plotting

        Args:
            external (bool, optional): True to use external score. Defaults to False.
            external_score (_type_, optional): External score to be used. Only activate when external == True. Defaults to None.
            title (str, optional): plot title. Defaults to "Welch t-test volcano".
            show (bool, optional): True to show the plot. Defaults to True.
            saving (bool, optional): True to save the plot. Defaults to False.
            save_path (str, optional): The path to save plot. Only activate when saving == True. Defaults to "./output/images/".
        """
        log_fold = self.scores["log_fold_change"]
        log_p = self.scores["log_p_value"]
        # choose fold change > 2 and p value < 0.05 in log scale
        significant = np.logical_and(
            np.abs(log_fold) >= np.log2(self.fc_threshold), log_p
            > -np.log10(self.p_threshold))

        if external:
            selected = pd.Series(False, index=self.scores.index)
            selected.loc[external_score.index] = True
        else:
            selected = pd.Series(False, index=significant.index)
            selected.loc[self.selected_score.index] = True

        # silent
        plt.scatter(x=log_fold[~significant],
                    y=log_p[~significant],
                    s=0.5,
                    color='gray')
        # not selected
        plt.scatter(x=log_fold[significant][~selected],
                    y=log_p[significant][~selected],
                    s=2)
        # selected
        if external:
            plt.scatter(x=log_fold[selected], y=log_p[selected], s=2)
        else:
            plt.scatter(x=log_fold[selected], y=log_p[selected], s=2)

        plt.title(title)
        plt.xlabel("log_2 fold")
        plt.axvline(1, linestyle="dotted", color="gray")
        plt.axvline(-1, linestyle="dotted", color="gray")
        plt.axhline(-np.log10(0.05), linestyle="dotted", color="gray")
        plt.ylabel("log_10 p")
        if saving:
            plt.savefig(save_path + title, format="png")
        plt.show()
