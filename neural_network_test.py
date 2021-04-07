from NeuralNetwork.neural_network import NeuralNetwork
from NeuralNetwork.layer import Layer
import numpy as np
from sklearn import datasets


def load_dataset():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    return X, y

def train_test_split(X, y):
    idx = np.arange(len(X))
    train_size = int(len(X) * 2 / 3)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':

    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    nn = NeuralNetwork()  # 实例化网络类
    nn.add_layer(Layer(4, 25, 'tanh'))  # 隐藏层1, 2=>25
    nn.add_layer(Layer(25, 3, 'sigmoid'))  # 隐藏层2, 25=>2
    learning_rate = 0.01
    max_epochs = 1000
    nn.train(X_train, X_test, y_train, y_test, learning_rate, max_epochs)