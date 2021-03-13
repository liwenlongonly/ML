# load required library
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
import scipy.optimize as opt

x, y = load_iris(return_X_y=True)
x = x[:100, :2]  # class 0 and 1 balanced
y = y[:100]

def draw_data(x, y):
    #########################################################################
    # Full Mark: 1                                                          #
    # TODO:                                                                 #
    # 1. make a scatter plot of the raw data                                #
    # 2. set title for the plot                                             #
    # 3. set label for x,y axis                                             #
    # Note, this scatter plot has two different type of points              #
    #########################################################################
    plt.title("Logistic Regression")
    plt.xlabel("x")
    plt.ylabel("y")
    pos = np.where(y == 0)
    neg = np.where(y == 1)
    plt.scatter(x[pos, 0], x[pos, 1], marker='x', color='r', label='class0')
    plt.scatter(x[neg, 0], x[neg, 1], marker='o', color='b', label='class1')
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
    y_ = np.dot(X, theta)
    s = 1.0 / (1 + np.exp(-y_))
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################

    return s

# define cost function with sigmoid function
def cost(theta, X, y):
    #########################################################################
    # Full Mark: 2                                                          #
    # TODO:                                                                 #
    # 1. implement the cross entropy loss function with sigmoid             #
    #########################################################################
    hx = sigmoid(theta, X)
    if np.sum(1 - hx < 1e-10) != 0:
        return np.inf
    co = -np.mean(np.multiply(y, np.log(hx)) + np.multiply(1 - y, np.log(1 - hx)))
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
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    return grad

# draw decision boudary
def draw_decision_boudary(final_theta,x,y):
    #########################################################################
    # Full Mark: 2                                                          #
    # TODO:                                                                 #
    # 1. plot the decision boudary on the raw data                          #
    # 2. set title for the plot                                             #
    # 3. set label for x,y axis                                             #
    # Note, this scatter plot has two different type of points              #
    #########################################################################
    plt.title("Logistic Regression")
    plt.xlabel("x")
    plt.ylabel("y")
    pos = np.where(y == 0)
    neg = np.where(y == 1)
    plt.scatter(x[pos, 1], x[pos, 2], marker='x', color='r', label='class0')
    plt.scatter(x[neg, 1], x[neg, 2], marker='o', color='b', label='class1')

    plot_x = np.array([np.min(x[:, 1]) - 1, np.max(x[:, 1] + 1)])
    plot_y = -1 / final_theta[2] * (final_theta[1] * plot_x + final_theta[0])

    plt.plot(plot_x, plot_y)

    plt.legend()
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################

    # show plot
    plt.show()


# predict for new X
def predict(theta, X):
    #########################################################################
    # Full Mark: 1                                                          #
    # TODO:                                                                 #
    # 1. predict the value using theta and sigmoid                          #
    # 2. convert the predicted value to 0/1                                 #
    # That's how it is called Logistic regression                           #
    #########################################################################
    m = len(X)
    predict_labels = np.zeros((m,))
    pos = np.where(sigmoid(theta, X) >= 0.5)
    neg = np.where(sigmoid(theta, X) < 0.5)
    predict_labels[pos] = 1
    predict_labels[neg] = 0
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################

    return predict_labels


# calculate accuracy
def accurate(predictions, y):
    #########################################################################
    # Full Mark: 1                                                          #
    # TODO:                                                                 #
    # 1. calculate the accuracy value                                       #
    # Note that you coud not import extra library                           #
    #########################################################################
    accuracy_score = np.sum(predictions == y) / len(y)
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################
    return accuracy_score


if __name__ == '__main__':
    draw_data(x, y)

    x = np.concatenate((np.array([np.ones(len(y))]).T, x), axis=1)
    theta = np.zeros(x.shape[1])

    result = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(x, y))
    final_theta = result[0]
    final_cost = cost(final_theta, x, y)
    predictions = predict(final_theta, x)
    accuracy = accurate(predictions, y)
    print("final cost is " + str(final_cost))
    print("accuracy is " + str(accuracy))

    draw_decision_boudary(final_theta, x, y)




