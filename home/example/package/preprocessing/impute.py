from sklearn.impute import KNNImputer, SimpleImputer
import pandas as pd
import numpy as np

class imputer():
    def __init__(self, threshold, center = True, scale = True, log_domain = True):
        # threshold sould between 0 ~ 1
        if 0 < threshold <1 or threshold == 1:
            self.threshold = threshold
        else:
            raise ValueError("missing value threshold must be a float from (0, 1]: ", threshold)
        self.normalizer = normalizer(center = center, scale = scale, log_domain = log_domain)
        self.fitted = False
    
    def fit(self, x, y = None):
        # drop too empty features
        self.not_too_empty = x.isna().mean() <= self.threshold
        x = x.loc[:, self.not_too_empty] # keep those who not too empty

        # normalize
        x, y = self.normalizer.fit_transform(x, y)

        # call the kernel
        self.kernel.fit(x)
        self.fitted = True
        return self

    def transform(self, x, y = None):
        if not self.fitted:
            raise "please call fit before calling transform."
        # drop too empty features
        x = x.loc[:, self.not_too_empty] # keep those who not too empty

        columns = x.columns
        idx = x.index

        # normalize
        x, y = self.normalizer.fit_transform(x, y)

        # call the kernel
        x = self.kernel.transform(x)

        # rebuild the dataframe from numpy array returns
        x = pd.DataFrame(x, columns = columns, index = idx)
        
        # inverse normalize
        x, y = self.normalizer.inverse_transform(x, y)
        return x, y

    def fit_transform(self, x, y = None):
        self.fit(x, y)
        x, y = self.transform(x, y)
        return x, y


class knn_imputer(imputer):
    def __init__(self, threshold = 0.3333, n_neighbor = 5):
        super().__init__(threshold)
        
        self.kernel = KNNImputer(n_neighbors = n_neighbor)


class simple_imputer(imputer):
    def __init__(self, threshold = 0.3333, strategy = "median"):
        super().__init__(threshold)

        self.kernel = SimpleImputer(strategy= strategy)


class normalizer:
    def __init__(self, center = True, scale = True, log_domain = False):
        self.center = center
        self.mean = 0
        self.scale = scale
        self.norm = 1
        self.log_domain = log_domain
        self.fitted = False

    def fit(self, x, y = None):
        if self.log_domain:
            x = np.log(x)
        if self.center:
            self.mean = x.mean()
        if self.scale:
            self.norm = x.std()
        self.fitted = True
        return self
    
    def transform(self, x, y = None):
        if not self.fitted:
            print("WARNING: imputer not initialized, please call fit before calling transform")
        if self.log_domain:
            x = np.log(x)
        if self.center:
            x = x - self.mean
        if self.scale:
            x = x / self.norm
        return x, y

    def fit_transform(self, x, y = None):
        self.fit(x, y)
        return self.transform(x, y)

    def inverse_transform(self, x, y = None):
        if self.scale:
            x = x * self.norm
        if self.center:
            x = x + self.mean
        if self.log_domain:
            x = np.exp(x)

        return x, y
    


if __name__ == "__main__":
    imputer(0.5)
    knn_imputer(0.5)
    simple_imputer(0.5)
