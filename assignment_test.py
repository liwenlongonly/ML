from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
import numpy as np

def load_dataset():
    '''
    ** Do not modify this function. **
    Load diabetes dataset. We only use one feature and 60 instances.
    '''

    X, y = load_diabetes(return_X_y=True)
    return X[:60, 2], y[:60]

def plot_data(X, y):
    '''
    Draw scatter plot using raw data.
    '''

    #########################################################################
    # Full Mark: 10                                                         #
    # TODO:                                                                 #
    # 1. make a scatter plot of the raw data                                #
    # 2. set title for the plot                                             #
    # 3. set label for X,y axis                                             #
    # e.g.,                                                                 #
    #https://matplotlib.org/3.2.0/api/_as_gen/matplotlib.pyplot.scatter.html#
    #########################################################################
    plt.title("Linear Regression Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(X, y, "ob")

    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################

    # return the plt object
    return plt

def train_test_split(X, y):
    '''
    Randomly split data into train and test set.
    Two thirds of the raw data will be the train set and one third of the raw data will be the test set.
    '''

    ######################################################################################
    # Full Mark: 10                                                                      #
    # TODO:                                                                              #
    # 1. shuffle the indices of data first.                                              #
    # (Hint: use numpy.arange and numpy.random.shuffle)                                  #
    # 2. select two thirds of the data as train set, the rest of data as test set.       #
    ######################################################################################

    X[:, 0] = y
    np.random.shuffle(X)
    y_train = np.array(X[:40, 0])
    y_test = np.array(X[40:, 0])
    X[:, 0] = np.ones(X.shape[0])
    X_test = X[40:]
    X_train = X[:40]
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################

    return X_train, X_test, y_train, y_test


def cost_function(weights, X, y):
    '''
    Define the cost function.
    '''

    #########################################################################
    # Full Mark: 25                                                         #
    # TODO:                                                                 #
    # Implement the Mean Square Error function:                             #
    # https://en.wikipedia.org/wiki/Mean_squared_error#Mean                 #
    #                                                                       #
    # (Hint: Use numpy functions)                                           #
    #########################################################################
    y_ = weights * X
    y_ = y_[:, 1] + y_[:, 0]
    cost = 1 / X.shape[0] * np.sum(np.square(y - y_))
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################

    # return cost
    return cost

def gradient_descent(weights, X, y):
    '''
    Update weights using gradient descent algorithm.
    '''

    # define your learning_rate and epoch
    lr = 0.1
    epoch = 20000

    # define cost
    cost_list = []

    # for loop
    for i in range(epoch):
        #########################################################################
        # Full Mark: 25                                                         #
        # TODO:                                                                 #
        # 1. update weights with learning rate lr                               #
        # 2. append the updated cost to cost list                               #
        # (Hint: Use numpy functions)                                           #
        #########################################################################
        cost = cost_function(weights, X, y)
        if len(cost_list) > 0 and cost > cost_list[-1]:
            break
        cost_list.append(cost)
        # g_w = 2 / X.shape[1] * np.sum(weights * X[1] * X[1] - y * X[1] + weights*X[1])
        # g_b = 2 / X.shape[1] * np.sum(weights - y + weights * X[1])
        # weights = weights - lr * g_w
        # X[0] = X[0] - lr * g_b
        y_ = weights * X
        y_ = y_[:, 1] + y_[:, 0]
        dw = (1.0 / X.shape[0]) * ((y_ - y) * X[:, 1]).sum()
        db = (1.0 / X.shape[0]) * ((y_ - y).sum())
        # weight
        weights[1] = weights[1] - lr * dw
        # bias
        weights[0] = weights[0] - lr * db
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    # return updated weights and cost list
    return weights, cost_list


def plot_iteration(cost, epoch=20000):
    '''
    Plot the cost for each iteration.
    '''

    #########################################################################
    # Full Mark: 10                                                         #
    # TODO:                                                                 #
    # 1. plot the cost for each iteration                                   #
    # 2. set title and labels for the plot                                  #
    # (Hint: Use plt.plot function to plot and range(n))                    #
    #########################################################################
    plt.title("cost for each iteration")
    plt.xlabel("epoch")
    plt.ylabel("cost")
    plt.plot(range(len(cost)), cost)
    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################

    # show plot
    plt.show()

def plot_final(weights, X, y):
    '''
    Draw the simple linear regression model.
    '''

    # draw the raw data first
    model_plot = plot_data(X, y)

    #########################################################################
    # Full Mark: 10                                                         #
    # TODO:                                                                 #
    # 1. create a series of x coordinates in proper range.                  #
    # (Hint: use numpy.arange)                                              #
    # 2. calculate y coordinates:                                           #
    #                         y = w * X + b                                 #
    # 3. plot the curve and set title                                       #
    #########################################################################
    y_ = weights[1] * X + weights[0]
    model_plot.plot(X, y_)

    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################

    # show plot
    model_plot.show()

def print_test_error(weights, X, y_true):
    '''
    Use optimized weights to predict y, and print test error.
    '''

    #########################################################################
    # Full Mark: 10                                                         #
    # TODO:                                                                 #
    # 1. predict the target value y of X:                                   #
    #                            y = w * X + b                              #
    # 2. calculate the Mean Square Error using true y and predicted y       #
    #########################################################################

    y_ = weights[1] * X + weights[0]
    error = 1 / X.shape[0] * np.sum(np.square(y_ - y_true))

    #########################################################################
    #                       END OF YOUR CODE                                #
    #########################################################################

    # print test error
    print("Test error: %.4f" % error)
    return error

def main():
    # Plot raw data points
    X, y = load_dataset()
    plot = plot_data(X, y)
    plot.show()

    # Split train and test set
    X = np.c_[np.ones(X.size), X]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print("X_train:")
    print(X_train)
    print("y_train:")
    print(y_train)
    print("X_test")
    print(X_test)
    print("y_test")
    print(y_test)

    # initialize weight
    weights = np.ones(X_train.shape[1])
    # calculate training cost
    init_cost = cost_function(weights, X_train, y_train)
    print("Initial cost: %.4f" % init_cost)

    # gradient descent to find the optimal fit
    weights, cost_list = gradient_descent(weights, X_train, y_train)
    print(weights)
    print(cost_list)

    # draw the cost change for iterations
    plot_iteration(cost_list)

    # draw the final linear model
    # it is shown as a red line, you can change the color anyway
    plot_final(weights, X_train[:, 1], y_train)

    # Print test error
    print_test_error(weights, X_test[:, 1], y_test)


if __name__ == '__main__':
    main()
