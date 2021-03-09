# load required library
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
import scipy.optimize as opt

x, y = load_iris(return_X_y=True)
# print(x)
x = x[:100, :2]  # class 0 and 1 balanced
y = y[:100]
# print(y)

def draw_data(x, y):
    #########################################################################
    # Full Mark: 1                                                          #
    # TODO:                                                                 #
    # 1. make a scatter plot of the raw data                                #
    # 2. set title for the plot                                             #
    # 3. set label for x,y axis                                             #
    # Note, this scatter plot has two different type of points              #
    #########################################################################
    plt.title("Title")
    plt.xlabel("x")
    plt.ylabel("y")
    class0_x = []
    class0_y = []
    class1_x = []
    class1_y = []
    for i, val in enumerate(y):
        if val == 0:
            class0_x.append(x[i][0])
            class0_y.append(x[i][1])
        else:
            class1_x.append(x[i][0])
            class1_y.append(x[i][1])

    plt.scatter(class0_x, class0_y, c='r', marker='x', label='class0')
    plt.scatter(class1_x, class1_y, c='b', marker='o', label='class1')
    plt.legend()
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################

    # show plot
    plt.show()

# define sigmoid function
# math: refer to https://en.wikipedia.org/wiki/Sigmoid_function or slides
def sigmoid(theta, X):
    #########################################################################
    # Full Mark: 1                                                          #
    # TODO:                                                                 #
    # 1. implement the sigmoid function over input theta and X
    #########################################################################
    y_ = np.sum(X * theta, axis=1)
    s = 1 / (1 + np.exp(-y_))
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################

    return s

def cross_entropy_error(y, t):
    return -np.sum(t * np.log(y))

# define cost function with sigmoid function
def cost(theta, X, y):
    #########################################################################
    # Full Mark: 2                                                          #
    # TODO:                                                                 #
    # 1. implement the cross entropy loss function with sigmoid             #
    #########################################################################
    y_ = sigmoid(theta, X)
    co = cross_entropy_error(y_, y)
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    return co


# the gradient of the cost is a vector of the same length as θ where the jth element (for j = 0, 1, . . . , n)
def gradient(theta, X, y):
    #########################################################################
    # Full Mark: 2                                                          #
    # TODO:                                                                 #
    # 1. calculate the gradients using theta and sigmoid                    #
    # Hint: X may need to be transposed to do matrix operation              #
    #########################################################################
    y_ = sigmoid(theta, X)
    grad = np.dot(X.T,  y_ - y)/len(X)
    # print(grad)
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    return grad


if __name__ == '__main__':
    draw_data(x, y)

    x = np.concatenate((np.array([np.ones(len(y))]).T, x), axis=1)
    theta = np.zeros(x.shape[1])

    result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(x, y))
    print(result)
    final_theta = result[0]
    final_cost = cost(final_theta, x, y)
    print(final_cost)


