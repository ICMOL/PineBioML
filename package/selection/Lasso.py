from .base import selection as base_selection
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import Lasso


def find_alpha(line):
    i = len(line)-1
    alive =  line==0
    while alive[i]:
        i-=1
    return i

def sample_weight(y):
    p = y.mean()
    q = 1-p
    sp = 1/p/(1/p+1/q)
    sq = 1/q/(1/p+1/q)
    return y*sp + (1-y)*sq

class Lasso_selection(base_selection):
    def __init__(self, balanced = False, center = True, scale = False):
        super().__init__(center = center, scale = scale, global_scale = True)
        
        # parameters
        self.da = 0.01 # d alpha
        self.blind = True
        self.upper_init = 10
        self.balanced = balanced
        self.name = "LassoLinear"

    def scoring(self, x, y = None):
        # kfold
        # (blank)

        # train test split
        if not self.blind:
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        else:
            X_train = x
            y_train = y

        if self.balanced:
            weights = np.ones_like(y_train)
        else:
            weights = sample_weight(y_train)

        lassoes = []
        score = []
        # grid searching
        for alpha in tqdm(np.arange(self.da, self.upper_init, self.da)):
            lassoes.append(Lasso(alpha=alpha))
            lassoes[-1].fit(X_train, y_train, weights)
            alive = (lassoes[-1].coef_ != 0).sum()
            
            if not self.blind:
                score.append(((lassoes[-1].predict(X_test)>0.5)== y_test).mean())

            if alive <1:
                print("all coefficient are dead, terminated.")
                break

        coef = np.array([clr.coef_ for clr in lassoes])

        self.scores = pd.Series(np.logical_not(coef == 0).sum(axis = 0)*self.da, index = x.columns, name = self.name).sort_values()
        return self.scores.copy()


class Lasso_bisection_selection(base_selection):
    def __init__(self, center = True, scale = False, balanced = False):
        super().__init__(center = center, scale = scale, global_scale = True)
        self.upper_init = 10
        self.lower_init = 1e-3
        self.balanced = balanced
        self.blind = True
        self.name = "LassoLinear"

    def select(self, x, y, k):
        # kfold
        # (blank)

        # normalize
        x, y = self.normalizer.fit_transform(x, y)

        # train test split
        if not self.blind:
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        else:
            X_train = x
            y_train = y

        if self.balanced:
            weights = np.ones_like(y_train)
        else:
            weights = sample_weight(y_train)

        lassoes = []
        score = []
        # Bisection searching
        upper = self.upper_init
        lassoes.append(Lasso(alpha=upper))
        lassoes[-1].fit(X_train, y_train, weights)
        if not self.blind:
            score.append(((lassoes[-1].predict(X_test)>0.5)== y_test).mean())
        upper_alive = (lassoes[-1].coef_ != 0).sum()
        #print(upper, upper_alive)
        
        lower = self.lower_init
        lassoes.append(Lasso(alpha=lower))
        lassoes[-1].fit(X_train, y_train, weights)
        if not self.blind:
            score.append(((lassoes[-1].predict(X_test)>0.5)== y_test).mean())
        lower_alive = (lassoes[-1].coef_ != 0).sum()
        #print(lower, lower_alive)
        
        while not lower_alive== k:
            alpha = (upper+lower)/2
            lassoes.append(Lasso(alpha=alpha))
            lassoes[-1].fit(X_train, y_train, weights)
            if not self.blind:
                score.append(((lassoes[-1].predict(X_test)>0.5)== y_test).mean())
            alive = (lassoes[-1].coef_ != 0).sum()
            #print(alpha, alive)
            if alive>= k:
                lower = alpha
                lower_alive = alive
            else:
                upper = alpha
                upper_alive = alive

        coef = np.array([clr.coef_ for clr in lassoes])
        
        self.scores = pd.Series(np.abs(coef[-1]), index = x.columns, name = self.name).sort_values()
        self.selected_score = self.scores.tail(k)
        return self.selected_score



class multi_Lasso_selection(base_selection):
    def __init__(self, center = True, scale = True):
        super().__init__()
        self.center = center
        self.scale = scale
        self.name = "multi_Lasso"

    def select(self, x, y, k, batch_k= 5):
        result = []
        if k == -1:
            k = x.shape[0]
        for i in range(k//batch_k):
            result.append(Lasso_bisection_selection(center = self.center, scale = self.scale).select(x, y, k = batch_k))
            x = x.drop(result[-1].index, axis = 1)
        result = pd.concat(result)
        result = result-result.min()
        result.name = self.name

        self.selected_score = result.sort_values()
        return self.selected_score.copy()
            
        
