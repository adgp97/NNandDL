import numpy as np
import sklearn.datasets

def load_dataset():
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_regression(n_samples=1000000, n_features=1, noise=25, bias=400)
    return train_X, train_Y