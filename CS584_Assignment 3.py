from sklearn import datasets
from scipy.optimize import minimize
import numpy as np


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

def init_weights(num_in, num_out):
    '''
    :param num_in: the number of input units in the weight array
    :param num_out: the number of output units in the weight array
    '''

    # Note that 'W' contains both weights and bias, you can consider W[0, :] as bias
    W = np.zeros((1 + num_in, num_out))
    print("Oringnal W:", W)
    ###################################################################################
    # Full Mark: 1                                                                    #
    # TODO:                                                                           #
    # Implement Xavier/Glorot uniform initialization                                   #
    #                                                                                 #
    # Hint: you can find the reference here:                                          #
    # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotNormal   #
    ###################################################################################

    a = np.sqrt(6. / (num_in + num_out))
    W[0, :] = np.random.uniform(low=-a, high=a, size=[num_out, num_in])
    W[1:, :] = np.random.uniform(low=-a, high=a, size=[num_out])
    print("Init W:", W)

    ###################################################################################
    #                       END OF YOUR CODE                                          #
    ###################################################################################

    return W

def sigmoid(x):
    '''
    :param x: input
    '''

    ###################################################################################
    # Full Mark: 0.5                                                                  #
    # TODO:                                                                           #
    # Implement sigmoid function:                                                     #
    #                             sigmoid(x) = 1/(1+e^(-x))                           #
    ###################################################################################

    res = 1 / (1 + np.exp(-x))

    ###################################################################################
    #                       END OF YOUR CODE                                          #
    ###################################################################################

    return res

def tanh(x):
    '''
    :param x: input
    '''

    ###################################################################################
    # Full Mark: 0.5                                                                  #
    # TODO:                                                                           #
    # Implement tanh function:                                                        #
    #                     tanh(x) = (e^x-e^(-x)) / (e^x+e^(-x))                       #
    ###################################################################################

    res = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    ###################################################################################
    #                       END OF YOUR CODE                                          #
    ###################################################################################

    return res

def sigmoid_gradient(x):
    '''
    :param x: input
    '''

    ###################################################################################
    # Full Mark: 1                                                                    #
    # TODO:                                                                           #
    # Computes the gradient of the sigmoid function evaluated at x.                   #
    #                                                                                 #
    ###################################################################################

    s = sigmoid(x)
    grad = s * (1 - s)

    ###################################################################################
    #                       END OF YOUR CODE                                          #
    ###################################################################################

    return grad

def tanh_gradient(x):
    '''
    :param x: input
    '''

    ###################################################################################
    # Full Mark: 1                                                                    #
    # TODO:                                                                           #
    # Computes the gradient of the tanh function evaluated at x.                      #
    #                                                                                 #
    ###################################################################################

    t = tanh(x)
    grad = 1 - np.square(t)

    ###################################################################################
    #                       END OF YOUR CODE                                          #
    ###################################################################################

    return grad


def forward(W, X):
    '''
    :param W: weights (including biases) of the neural network. It is a list containing both W_hidden and W_output.
    :param X: input. Already added one additional column of all "1"s.
    '''

    # Shape of W_hidden: [num_feature+1, num_hidden]
    # Shape pf W_output: [num_hidden+1, num_output]
    W_hidden, W_output = W

    ###################################################################################
    # Full Mark: 1                                                                    #
    # TODO:                                                                           #
    # Implement the forward pass. You need to compute four values:                    #
    # z_hidden, a_hidden, z_output, a_output                                          #
    #                                                                                 #
    # Note that our neural network consists of three layers:                          #
    # Input -> Hidden -> Output                                                       #
    # The activation function in hidden layer is 'tanh'                               #
    # The activation function in output layer is 'sigmoid'                            #
    ###################################################################################

    z_hidden = np.sum(X * W_hidden, axis=1)
    a_hidden = tanh(z_hidden)

    z_output = np.sum(a_hidden * W_output, axis=1)
    a_output = sigmoid(z_output)

    ###################################################################################
    #                       END OF YOUR CODE                                          #
    ###################################################################################

    # z_hidden is the raw output of hidden layer, a_hidden is the result after performing activation on z_hidden
    # z_output is the raw output of output layer, a_output is the result after performing activation on z_output
    return {'z_hidden': z_hidden, 'a_hidden': a_hidden,
            'z_output': z_output, 'a_output': a_output}


def loss_funtion(W, X, y, num_feature, num_hidden, num_output, L2_lambda):
    '''
    :param W: a 1D array containing all weights and biases.
    :param X: input
    :param y: labels of X
    :param num_feature: number of features in X
    :param num_hidden: number of hidden units
    :param num_output: number of output units
    :param L2_lambda: the coefficient of regularization term
    '''

    m = y.size

    # Reshape W back into the parameters W_hidden and W_output
    W_hidden = np.reshape(W[:num_hidden * (num_feature + 1)],
                          ((num_feature + 1), num_hidden))

    W_output = np.reshape(W[(num_hidden * (num_feature + 1)):],
                          ((num_hidden + 1), num_output))

    # Initialize grads
    W_hidden_grad = np.zeros(W_hidden.shape)
    W_output_grad = np.zeros(W_output.shape)

    # Add one column of "1"s to X
    X_input = np.concatenate([np.ones((m, 1)), X], axis=1)

    ##########################################################################################
    # Full Mark: 3                                                                           #
    # TODO:                                                                                  #
    # 1. Transform y to one-hot encoding. Implement binary cross-entropy loss function       #
    # (Hint: Use the forward function to get necessary outputs from the model)               #
    #                                                                                        #
    # 2. Add L2 regularization to all weights in loss                                        #
    # (Note that we will not add regularization to bias)                                     #
    #                                                                                        #
    # 3. Compute the gradient for W_hidden and W_output (including both weights and biases)  #
    # (Hint: use chain rule, and don't forget to add the gradient of regularization term)    #
    ##########################################################################################

    regularization_loss = (L2_lambda / 2*m) * (np.sum(np.square(W_hidden[1:, :].ravel())) + np.sum(np.square(W_output[1:, :].ravel())))
    y_ = forward((W_hidden, W_output), X)
    z_hidden = y_["z_hidden"]
    a_hidden = y_["a_hidden"]
    z_output = y_["z_output"]
    a_output = y_["a_output"]
    y_onehot = np.zeros((y.shape[0], num_output))
    y_onehot[np.arange(y.shape[0]), y] = 1
    if np.sum(1 - a_output < 1e-10) != 0:
        return np.inf
    data_loss = -np.mean(np.multiply(y, np.log(a_output)) + np.multiply(1 - y, np.log(1 - a_output)))
    L = data_loss + regularization_loss

    W_hidden_grad = 0
    W_output_grad = 0

    ###################################################################################
    #                       END OF YOUR CODE                                          #
    ###################################################################################

    grads = np.concatenate([W_hidden_grad.ravel(), W_output_grad.ravel()])

    return L, grads

def optimize(initial_W, X, y, num_epoch, num_feature, num_hidden, num_output, L2_lambda):
    '''
    :param initial_W: initial weights as a 1D array.
    :param X: input
    :param y: labels of X
    :param num_epoch: number of iterations
    :param num_feature: number of features in X
    :param num_hidden: number of hidden units
    :param num_output: number of output units
    :param L2_lambda: the coefficient of regularization term
    '''

    options = {'maxiter': num_epoch}

    ###########################################################################################
    # Full Mark: 1                                                                            #
    # TODO:                                                                                   #
    # Optimize weights                                                                        #
    # (Hint: use scipy.optimize.minimize to automatically do the iteration.)                  #
    # ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) #
    # For some optimizers, you need to set 'jac' as True.                                     #
    # You may need to use lambda to create a function with one parameter to wrap the          #
    # loss_funtion you have implemented above.                                                #
    #                                                                                         #
    # Note that scipy.optimize.minimize only accepts a 1D weight array as initial weights,    #
    # and the output optimized weights will also be a 1D array.                               #
    # That is why we unroll the initial weights into a single long vector.                    #
    ###########################################################################################

    W_final = 0

    ###################################################################################
    #                       END OF YOUR CODE                                          #
    ###################################################################################

    # Obtain W_hidden and W_output back from W_final
    W_hidden = np.reshape(W_final[:num_hidden * (num_feature + 1)],
                          ((num_feature + 1), num_hidden))
    W_output = np.reshape(W_final[(num_hidden * (num_feature + 1)):],
                          ((num_hidden + 1), num_output))

    return [W_hidden, W_output]

def predict(X_test, y_test, W):
    '''
    :param X_test: input
    :param y_test: labels of X_test
    :param W: a list containing two weights W_hidden and W_output.
    '''

    test_input = np.concatenate([np.ones((y_test.size, 1)), X_test], axis=1)

    ###################################################################################
    # Full Mark: 1                                                                    #
    # TODO:                                                                           #
    # Predict on test set and compute the accuracy.                                   #
    # (Hint: use forward function to get predicted output)                            #
    #                                                                                 #
    ###################################################################################

    acc = 0

    ###################################################################################
    #                       END OF YOUR CODE                                          #
    ###################################################################################

    return acc


if __name__ == '__main__':
    # Define parameters
    NUM_FEATURE = 4
    NUM_HIDDEN_UNIT = 10
    NUM_OUTPUT_UNIT = 3
    NUM_EPOCH = 100
    L2_lambda = 1

    # Load data
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    # print(X_train)
    # print(y_train)
    #
    # print(X_test)
    # print(y_test)

    # # Initialize weights
    initial_W_hidden = init_weights(NUM_FEATURE, NUM_HIDDEN_UNIT)
    initial_W_output = init_weights(NUM_HIDDEN_UNIT, NUM_OUTPUT_UNIT)
    initial_W = np.concatenate([initial_W_hidden.ravel(), initial_W_output.ravel()], axis=0)
    print("initial_W:", initial_W)

    # Neural network learning
    # W = optimize(initial_W, X_train, y_train, NUM_EPOCH, NUM_FEATURE, NUM_HIDDEN_UNIT, NUM_OUTPUT_UNIT, L2_lambda)
    # print("W:", W)
    # Predict
    # acc = predict(X_test, y_test, W)
    # print("Test accuracy:", acc)
