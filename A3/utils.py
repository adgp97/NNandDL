import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets

def load_dataset_1():
    # Generate the dataset
    x = np.arange(0, 360 + 0.1, 0.1).reshape(-1,1) * np.pi / 180
    y = np.sin(x)
    return x, y

def load_dataset_2():
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    # Visualize the data
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral)

    return train_X, train_Y

def plot_decision_boundary(model, X, y):
    X = X.detach().numpy().T
    y = y.detach().numpy().reshape((1, -1))

    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    Z = model.forward(torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float())
    Z = Z.detach().numpy()
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y.ravel(), cmap=plt.cm.Spectral)
    plt.show()