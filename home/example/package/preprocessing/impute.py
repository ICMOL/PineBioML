from sklearn.impute import KNNImputer, SimpleImputer
import pandas as pd

class imputer():
    def __init__(self, threshold):
        # threshold sould between 0 ~ 1
        if 0 < threshold <1 or threshold == 1:
            self.threshold = threshold
        else:
            raise ValueError("missing value threshold must be a float from (0, 1]: ", threshold)

    def fit(self, x, y = None):
        return self

    def transform(self, x, y = None):
        return x, y

    def fit_transform(self, x, y = None):
        self.fit(x, y)
        return self.transform(x, y)


class knn_imputer(imputer):
    def __init__(self, threshold = 0.3333, n_neighbor = 5):
        super().__init__(threshold)

        self.kernel = KNNImputer(n_neighbors = n_neighbor)
        self.mean = None
        self.std = None
        self.fitted = False

    def fit(self, x, y = None):
        # drop too empty features
        self.not_too_empty = x.isna().mean() <= self.threshold
        x = x.loc[:, self.not_too_empty] # keep those who not too empty
        
        self.mean = x.mean()
        self.std = x.std()

        # normalize before kernel
        x = (x-self.mean)/self.std

        # call the kernel
        self.kernel.fit(x)
        self.fitted = True
        return self

    def transform(self, x, y = None):
        if not self.fitted:
            raise "please call fit before calling transform."
        # drop too empty features
        x = x.loc[:, self.not_too_empty] # keep those who not too empty

        # normalize before kernel
        x = (x-self.mean)/self.std
        columns = x.columns
        idx = x.index

        # call the kernel
        x = self.kernel.transform(x)

        # back to the original scale
        x = x*self.std.to_numpy().reshape(1, -1) +self.mean.to_numpy().reshape(1, -1)
        x = pd.DataFrame(x, columns = columns, index = idx)
        
        return x, y


class simple_imputer(imputer):
    def __init__(self, threshold = 0.3333, strategy = "median"):
        super().__init__(threshold)

        self.kernel = SimpleImputer(strategy= strategy)

    def fit(self, x, y = None):
        # call the kernel
        self.kernel.fit(x)
        
        return self

    def transform(self, x, y = None):
        # call the kernel
        x = self.kernel.transform(x)
        return x, y

    

if __name__ == "__main__":
    imputer(0.5)
    knn_imputer(0.5)
    simple_imputer(0.5)
