import numpy as np
import scipy.optimize as opt
from sklearn import datasets

input_layer_size = 4
hidden_layer_size = 10
num_labels = 3

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

def sigmoid(x):
    res = 1 / (1 + np.exp(-x))
    return res

def tanh(x):
    res = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return res

def sigmoid_gradient(x):
    s = sigmoid(x)
    grad = s * (1 - s)
    return grad

def tanh_gradient(x):

    h = tanh(x)
    grad = 1 - np.square(h)
    return grad

def feed_forward(theta, X):
    '''得到每层的输入和输出'''
    t1, t2 = deserialize(theta)   # 提取参数 t1是第一层到第二层的  t2是第二层到第三层的
    a1 = X   #初始值
    z2 = a1 @ t1.T   # X乘参数
    a2 = np.insert(tanh(z2), 0, 1, axis=1)  #加a0 并且放入sigmoid函数中
    z3 = a2 @ t2.T   #第二层到第三层
    a3 = sigmoid(z3)
    return a1, z2, a2, z3, a3

def cost(theta, X, y):
    a1, z2, a2, z3, h = feed_forward(theta, X)#前馈神经网络 第一层401个单元 第二层26个单元 第三层10个单元
    J = - y * np.log(h) - (1 - y) * np.log(1 - h)    #矩阵点乘
    return J.sum() / len(X)

def regularized_cost(theta, X, y, l=1):
    '''正则化时忽略每层的偏置项，也就是参数矩阵的第一列'''
    t1, t2 = deserialize(theta)
    reg = np.sum(t1[:, 1:] ** 2) + np.sum(t2[:, 1:] ** 2)    # 正则项
    loss = l / (2 * len(X)) * reg + cost(theta, X, y)  # 代价函数
    # print("loss:", loss)
    return loss

def deserialize(seq):
    '''
    提取参数
    '''
    hidden_size = hidden_layer_size
    input_size = input_layer_size + 1
    return seq[:hidden_size*input_size].reshape(hidden_size, input_size), seq[hidden_size*input_size:].reshape(num_labels, hidden_size+1)

def serialize(a, b):
    '''
    展开参数
    '''
    return np.r_[a.flatten(),b.flatten()]

def random_init(size):
    '''从服从的均匀分布的范围中随机返回size大小的值'''
    return np.random.uniform(-0.12, 0.12, size)


def gradient(theta, X, y):
    '''
    unregularized gradient, notice no d1 since the input layer has no error
    return 所有参数theta的梯度，故梯度D(i)和参数theta(i)同shape，重要。
    '''
    t1, t2 = deserialize(theta) # t1:(5, 10) t2:(11, 3)
    a1, z2, a2, z3, h = feed_forward(theta, X)
    # a1:(100, 5) z2:(100, 10) a2:(100, 11) z3:(100, 3)
    d3 = h - y  # (150, 3)
    d2 = d3 @ t2[:, 1:] * tanh_gradient(z2)  # (100, 10)
    D2 = d3.T @ a2  # (3, 11)
    D1 = d2.T @ a1  # (10, 5)
    D = (1 / len(X)) * serialize(D1, D2)  # (83,)

    return D

def regularized_gradient(theta, X, y, l=1):
    """
    不惩罚偏置单元的参数   正则化神经网络
    """
    t1, t2 = deserialize(theta)
    D1, D2 = deserialize(gradient(theta, X, y))
    t1[0, :] = 0
    t2[0, :] = 0
    reg_D1 = D1 + (l / len(X)) * t1
    reg_D2 = D2 + (l / len(X)) * t2
    return serialize(reg_D1, reg_D2)

def nn_training(X, y):
    size = hidden_layer_size * (input_layer_size + 1) + num_labels * (hidden_layer_size + 1)
    init_theta = random_init(size) #

    res = opt.minimize(fun=regularized_cost,
                       x0=init_theta,
                       args=(X, y, 1),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'maxiter': 400})
    return res

def accuracy(theta, X, y):
    _, _, _, _, h = feed_forward(res.x, X)
    pred = np.argmax(h, axis=1)
    print("pred:", pred)
    acc = np.sum(pred == y) / len(y)
    print("acc:", acc)


if __name__ == '__main__':

    raw_X, raw_y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(raw_X, raw_y)
    X = np.insert(X_train, 0, 1, axis=1)  # 加一列 1 (100, 5)
    y = np.zeros((y_train.shape[0], num_labels))
    y[np.arange(y_train.shape[0]), y_train] = 1

    # print("X:", X)
    # print("y:", y)

    res = nn_training(X, y)  # 慢
    print(res)

    X_ = np.insert(X_test, 0, 1, axis=1)
    # ——————————————4. 检验——————————————————
    accuracy(res.x, X_, y_test)
