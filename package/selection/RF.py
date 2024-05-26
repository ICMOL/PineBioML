from .base import selection as base_selection
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.decomposition import PCA, NMF
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
from multiprocessing import Pool

class RF_selection(base_selection):
    def __init__(self, trees = 1024*16, unbalanced = True, strategy = "gini", center = True, scale = True):
        super().__init__(center = center, scale = scale)
        self.strategy = strategy
        if unbalanced:
            class_weight = "balanced"
        else:
            class_weight = None
            
        self.kernel = RandomForestClassifier(n_estimators = trees, n_jobs=-1, max_samples = 0.75, class_weight = class_weight, criterion = strategy, verbose = 1, ccp_alpha=1e-2)
        self.name = "RandomForest_"+self.strategy

    def scoring(self, x, y = None):
        self.kernel.fit(x, y)
        score = self.kernel.feature_importances_
        self.scores = pd.Series(score, index=x.columns, name = self.name).sort_values(ascending = False)
        return self.scores.copy()



class oob_RFClassifier:
    def __init__(self, trees = 1024*8, unbalanced = True, strategy = "gini"):
        self.strategy = strategy
        if unbalanced:
            class_weight = "balanced"
        else:
            class_weight = None
        self.subsampling_ratio = 0.7
        self.n_trees = trees
        self.tree_parms = {
            "criterion": strategy,
            "splitter": 'best',
            "max_depth": None, 
            "min_samples_split": 2, 
            "min_samples_leaf": 1, 
            "min_weight_fraction_leaf": 0.0, 
            "max_features": "sqrt", 
            "class_weight": class_weight, 
            "ccp_alpha": 1e-2
        }
        self.trees = {}

    def fit(self, x, y):
        self.subsampling_table = pd.DataFrame(np.random.binomial(1, self.subsampling_ratio, size = (x.shape[0], self.n_trees)).astype(np.bool_), index = x.index)
        for i_th in tqdm(self.subsampling_table.columns):
            subsample = self.subsampling_table[i_th]
            sub_x = x.loc[subsample]
            sub_y = y.loc[subsample]

            self.trees[i_th] = DecisionTreeClassifier(**self.tree_parms).fit(sub_x, sub_y)

    def predict_prob(self, x):
        return pd.Series([t.predict(x) for t in self.trees.items], index= x.index, name = "RF_prob").mean(axis = 1)

    def oob_predict_prob(self, x):
        if len(x)==len(self.subsampling_table):
            if not (x.index == self.subsampling_table.index).all(): 
                print("oob_predict_prob detect input x which diffirs from training")
        else:
            print("oob_predict_prob detect input x which diffirs from training")

        predicts = pd.Series([t.predict(x) for t in self.trees.items], index= x.index, name = "RF_prob").divide
        oob_mask = np.logical_not(self.subsampling_table)
        predicts = predicts*oob_mask
        
        return predicts.sum(axis = 1)/oob_mask.sum(axis = 1)

    def evaluate(self, x, y, metric = "ACC", oob = False):
        if oob:
            y_pred = self.oob_predict_prob(x)
        else:
            y_pred = self.predict_prob(x)

        if metric in ["ACC" , "acc" , "accuracy"]:
            return accuracy_score(y, y_pred>0.5) 
        elif metric in ["f1", "F1", "f1_score", "F1_score"]:
            return f1_score(y, y_pred>0.5)
        elif metric in ["BCE", "bce", "cross_entropy", "log_loss"]:
            return log_loss(y, y_pred)
        elif metric in ["AUC", "auc", "roc_auc", "ROC_AUC"]:
            return roc_auc_score(y, y_pred)
        else:
            print("Metric ",metric ," not support! Please use one of acc, f1, bce or roc_auc.")
            return 0
    
        
class pcRF_selection(base_selection):
    def __init__(self, trees = 512, unbalanced = True, strategy = "permutation", factorize_method = "PCA", center = True, scale = True):
        super().__init__(center = center, scale = scale)
        # remove colinearity
#        if factorize_method == "NMF":
 #           self.fatorizer = NMF()
  #      else:
        #   self.fatorizer = PCA()

        if unbalanced:
            class_weight = "balanced"
        else:
            class_weight = None
        
        self.strategy = strategy
        #self.kernel = RandomForestClassifier(n_estimators = trees, bootstrap=True, oob_score=True, n_jobs=-1, class_weight = class_weight, verbose = 1)
        self.kernel = oob_RFClassifier(trees)
        self.name = "pcRandomForest_"+self.strategy

    def scoring(self, x, y = None):
        print("pc RF kernel fitting.")
        self.kernel.fit(x, y)

        n_repeat = 5
        print("Permutation evaluation start! n_repeat: ", n_repeat)
        score = pd.DataFrame(0, index = x.columns, columns = [i for i in range(n_repeat)])
        for i in range(n_repeat): # repeat 5 times
            print("    repeat: ", i)
            for col in tqdm(x.columns): # permutate each column
                x_permute = x.copy()
                x_permute.loc[:, col] = shuffle(x_permute[col]).values
                score.loc[col, i] = self.kernel.evaluate(x_permute, y, oob = True)
        
        score = score.mean(axis = 1)
        
        return score
    
    def select(self, x, y, k):
        # x should be a pd dataframe or a numpy array without missing value
        x, y = self.normalizer.fit_transform(x, y)
        columns = x.columns
        index = x.index

        # remove colinearity
        # tune number of factor
        tmpca = PCA()
        pc_x = tmpca.fit_transform(x)
        accumucate_evr = 0
        threshold = 0.95
        i = 0
        for evr in tmpca.explained_variance_ratio_:
            accumucate_evr+=evr
            i+=1
            if accumucate_evr>threshold:
                break
        self.n_pc = i

        # do the decomposition
        self.factorizer = PCA(self.n_pc)
        pc_x = pd.DataFrame(self.factorizer.fit_transform(x), index= index)

        # scoring
        pc_scores = self.scoring(pc_x, y)

        # revert importance of pricipal component to importance of feature via ratio of variance
        self.scores = pd.Series(
            pc_scores.dot(np.abs(self.factorizer.components_*self.factorizer.explained_variance_ratio_.reshape(self.n_pc, 1))),#
            index = columns,
            name = self.name
        ).sort_values(ascending = False)
        
        selected_score = self.choose(self.scores, k)
        return selected_score        

class AdaBoost_selection(base_selection):
    def __init__(self, unbalanced = True, n_iter = 64, learning_rate = 0.1, center = True, scale = True):
        super().__init__(center = center, scale = scale)
        self.unbalanced = unbalanced
        self.kernel = AdaBoostClassifier(n_estimators = n_iter, learning_rate= learning_rate)
        self.name = "AdaBoost"+str(n_iter)

    def scoring(self, x, y = None):
        print("I don't have a progress bar but I am running")
        if self.unbalanced:
            sample_weight = (y/y.mean() + (1-y)/((1-y).mean()))/2
        else:
            sample_weight = np.ones(len(y))
        self.kernel.fit(x, y, sample_weight)
        score = self.kernel.feature_importances_
        self.scores = pd.Series(score, index=x.columns, name = self.name).sort_values(ascending = False)
        return self.scores.copy()

if __name__ == "__main__":
    RF_selection()
