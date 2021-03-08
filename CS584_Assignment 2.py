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


if __name__ == '__main__':
    draw_data(x, y)
