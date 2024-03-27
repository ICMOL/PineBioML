import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import Lasso
from .base import selection as base_selection

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
    def __init__(self, balanced = False, plotting = False, center = True, scale = False, log_transform = True):
        super().__init__(center = center, scale = scale, log_transform = log_transform)
        
        # parameters
        self.da = 0.01 # d alpha
        self.blind = True
        self.upper_init = 40
        self.balanced = balanced
        self.plotting = plotting

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
        lifetime = []
        # grid searching
        for alpha in tqdm(np.arange(1e-3, self.upper_init, self.da)):
            lassoes.append(Lasso(alpha=alpha))
            lassoes[-1].fit(X_train, y_train, weights)
            alive = (lassoes[-1].coef_ != 0).sum()
            lifetime.append(alive)
            
            if not self.blind:
                score.append(((lassoes[-1].predict(X_test)>0.5)== y_test).mean())

            if alive <1:
                print("all coefficient are dead, terminated.")
                break

        coef = np.array([clr.coef_ for clr in lassoes])
        lifetime = np.array(lifetime)
        score = [lifetime, coef, x.columns]

        return score

    def choose(self, score, k):
        lifetime = score[0]
        coef = score[1]
        names = score[2]


        idx = 0
        threshold = lifetime>k
        while threshold[idx]:
            idx += 1
        
        idx -= 1
        top_k = coef[idx] != 0
        coef_k = coef[:, top_k].T
        name_k = names[top_k]

        if self.plotting:
            for i in range(len(coef)):
                plt.plot(coef[i], label = name_20[i])

            plt.xlabel("alpha")
            plt.ylabel("coef")
            plt.legend()
            plt.show()

            plt.plot(score)
            plt.xlabel("alpha")
            plt.ylabel("test acc")
            plt.title("model accuracy")
            plt.show()
        
        LS_importance = pd.Series([find_alpha(i) for i in coef_k], index = name_k, name = "LassoLinear").sort_values()[-k:]*self.da
        return LS_importance


class Lasso_bisection_selection(base_selection):
    def __init__(self, center = True, scale = False, log_transform = True):
        super().__init__(center = center, scale = scale, log_transform = log_transform)
        self.upper_init = 40
        self.lower_init = 1e-3
        self.plotting = False
        self.balanced = False
        self.blind = True

    def select(self, x, y, k):
        # feature standarized
        print(x.values.std())
        x = x / x.values.std()

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
        
        top_k = coef[-1] != 0
        coef_k = coef[:, k].T
        name_k = x.columns[top_k]

        
        LS_importance = pd.Series(0, index = name_k, name = "LassoLinear")
        return LS_importance



class multi_Lasso_selection(base_selection):
    def __init__(self):
        super().__init__()
    def select(self, x, y, k, batch_k= 5):
        result = []
        for i in range(k//batch_k):
            result.append(Lasso_bisection_selection().select(x, y, k = batch_k))
            x = x.drop(result[-1].index, axis = 1)
        result = pd.concat(result)
        result.name = "multi_Lasso"
        return result-result.min()
            
        
