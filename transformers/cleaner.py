from sklearn.base import BaseEstimator, TransformerMixin
import seaborn as sns
import pandas as pd
import numpy as np

# This class cleans a pandas table and returns a numpy matrix.
# Pass an array of columns to drop. All null member columns will
# also be dropped.
class Kenformer(BaseEstimator, TransformerMixin):
    def __init__(self, drop_list=[]):
        self.drop_list = drop_list
        self.columns   = []

    # The fit(X, y[]) function is used by estimators to make
    # an estimate. This method returns "self".
    def fit(self, X, y=None):
        self.columns = X.columns.tolist()
        return self

    def transform(self, X):
        # 1. Drop columns declared in the drop_list
        X = X.drop(columns=self.drop_list)
        self.columns = X.columns.tolist()
        # 2. Drop rows containing null data
        X = X.dropna(how='any')
        # 3. Return as a numpy array
        return X.to_numpy()



if __name__ == "__main__":

    iris = sns.load_dataset('iris')
    print(type(iris))
    #print("This is how big the dataframe starts: ", str(len(df)))
    #kformer = Kenformer(['median_income'])
    #print(kformer.columns)
    #fitted_kformer = kformer.fit(df)
    #print(kformer.columns)
    #transformed_fitted_kformer = kformer.transform(df)
    #print("Now it's this long: ", str(len(transformed_fitted_kformer)))
    #print(kformer.columns)
    #print(transformed_fitted_kformer)
