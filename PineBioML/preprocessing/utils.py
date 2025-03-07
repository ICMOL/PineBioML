from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSSVD
import pandas as pd
import numpy as np

# Todo: sklearn.pipeline.FeatureUnion


class feature_extension():

    def __init__(self,
                 alpha=0.9,
                 pca=True,
                 pls=False,
                 cross=True,
                 ratio=True,
                 name="extend "):
        self.name = name
        self.alpha = alpha
        if pca:
            self.pca = PCA()
        else:
            self.pca = None

        if pls:
            self.pls = PLSSVD(4)
        else:
            self.pls = None

        self.cross = cross
        self.ratio = ratio

    def fit(self, x, y):
        results = [x.copy()]

        self.monotonic_feature = x.columns[np.logical_or(
            x.min() >= 0,
            x.max() <= 0)]
        self.monotonic_mean = x[self.monotonic_feature].mean()

        monotonic_x = x[self.monotonic_feature] / self.monotonic_mean

        if self.cross:
            x = monotonic_x.copy()
            column_names = monotonic_x.columns
            n = len(column_names)
            for i in range(n):
                for j in range(i + 1):
                    a = column_names[i]
                    b = column_names[j]
                    tmp = x[a] * x[b]
                    tmp.name = a + " * " + b

        if self.ratio:
            x = monotonic_x.copy()
            column_names = monotonic_x.columns
            n = len(column_names)
            for i in range(n):
                for j in range(i):
                    a = column_names[i]
                    b = column_names[j]
                    tmp = np.arctan2(x[a], x[b])
                    tmp.name = "arctan " + a + " / " + b

        x = results[0].copy()

        # PCA
        if self.pca:
            self.pca.fit(x)
            evr = self.pca.explained_variance_ratio_

            aev = 0
            counter = 0
            while aev < self.alpha and counter < len(evr):
                aev += evr[counter]
                counter += 1

            self.pca = PCA(counter)
            pcx = self.pca.fit_transform(x)
            pcx = pd.DataFrame(pcx,
                               index=x.index,
                               columns=[
                                   self.name + "pc" + str(i)
                                   for i in range(self.pca.n_components_)
                               ])

        if self.pls:
            if x.shape[1] < self.pls.n_components:
                self.pls.n_components = x.shape[1]
            self.pls.fit(x, y)

            plsx = self.pls.transform(x)
            plsx = pd.DataFrame(
                plsx,
                index=x.index,
                columns=[self.name + "pls" + str(i) for i in range(4)])

        return self

    def transform(self, x):
        results = [x.copy()]

        monotonic_x = x[self.monotonic_feature] / self.monotonic_mean

        if self.cross:
            x = monotonic_x.copy()
            column_names = monotonic_x.columns
            n = len(column_names)
            for i in range(n):
                for j in range(i + 1):
                    a = column_names[i]
                    b = column_names[j]
                    tmp = x[a] * x[b]
                    tmp.name = a + " * " + b
                    results.append(tmp)

        if self.ratio:
            x = monotonic_x.copy()
            column_names = monotonic_x.columns
            n = len(column_names)
            for i in range(n):
                for j in range(i):
                    a = column_names[i]
                    b = column_names[j]
                    tmp = np.arctan2(x[a], x[b])
                    tmp.name = "arctan " + a + " / " + b
                    results.append(tmp)

        x = results[0].copy()

        # PCA
        if self.pca:
            pcx = self.pca.transform(x)
            pcx = pd.DataFrame(pcx,
                               index=x.index,
                               columns=[
                                   self.name + "pc" + str(i)
                                   for i in range(self.pca.n_components_)
                               ])
            results.append(pcx)

        if self.pls:
            plsx = self.pls.transform(x)
            plsx = pd.DataFrame(
                plsx,
                index=x.index,
                columns=[self.name + "pls" + str(i) for i in range(4)])
            results.append(plsx)

        return pd.concat(results, axis=1)

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x)

    def report(self):
        state = {}
        return state
