from sklearn.base import BaseEstimator, TransformerMixin
import seaborn as sns
import pandas as pd
import numpy as np



if __name__ == "__main__":

    iris = sns.load_dataset('iris')
    print(type(iris))
