import numpy as np

def sigmoid(x):
    '''
    Activation function: f(x) = 1 / (1 + e^(-x))
    '''
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    '''
    Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    '''
    fx = sigmoid(x)
    return fx * (1 - fx)

def mse_loss(y_true, y_pred):
    '''
    y_true and y_pred are numpy arrays of the same length.
    '''
    return ((y_true - y_pred) ** 2).mean()

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def total(self, inputs):
        return np.dot(self.weights, inputs) + self.bias

    def feedforward(self, inputs):
        # Weight inputs, add bias, then use the activation function
        return sigmoid(self.total(inputs))

class NeuralNetwork:
    '''
    A neural network with:
    - 2 inputs
    - a hidden layer with 2 neurons (h1, h2)
    - an output layer with 1 neuron (o1)
    *** DISCLAIMER ***:
    The code below is intended to be simple and educational, NOT optimal.
    Real neural net code looks nothing like this. DO NOT use this code.
    Instead, read/run it to understand how this specific network works.
    '''
    def __init__(self):
        # Weights
        w1 = np.array([np.random.normal(), np.random.normal()])
        w2 = np.array([np.random.normal(), np.random.normal()])
        w3 = np.array([np.random.normal(), np.random.normal()])
        # Biases
        b1 = np.random.normal()
        b2 = np.random.normal()
        b3 = np.random.normal()

        self.h1 = Neuron(w1, b1)
        self.h2 = Neuron(w2, b2)
        self.o1 = Neuron(w3, b3)

    def feedforward(self, x):
        # x is a numpy array with 2 elements
        h1 = self.h1.feedforward(x)
        h2 = self.h2.feedforward(x)
        return self.o1.feedforward(np.array([h1, h2]))

    def train(self, data, y_trues):
        '''
        - data is a (n x 2) numpy array, n = # of samples in the dataset.
        - y_trues is a numpy array with n elements.
        Elements in y_trues correspond to those in data.
        '''
        learn_rate = 0.1
        epochs = 1000 # number of times loop through the entire dataset

        for epoch in range(epochs):
            for x, y_true in zip(data, y_trues):
                # do a feedforward
                sum_h1 = self.h1.total(x)
                h1 = self.h1.feedforward(x)

                sum_h2 = self.h2.total(x)
                h2 = self.h2.feedforward(x)

                h = np.array([h1, h2])
                sum_o1 = self.o1.total(h)
                y_pred = self.o1.feedforward(h)

                # calculate partial derivatives
                dL_dypred = -2 * (y_true - y_pred)

                # Neuron o1
                dypred_dw5 = h1 * deriv_sigmoid(sum_o1)
                dypred_dw6 = h2 * deriv_sigmoid(sum_o1)
                dypred_db3 = deriv_sigmoid(sum_o1)

                dypred_dh1 = self.o1.weights[0] * deriv_sigmoid(sum_o1)
                dypred_dh2 = self.o1.weights[1] * deriv_sigmoid(sum_o1)

                # Neuron h1
                dh1_dw1 = x[0] * deriv_sigmoid(sum_h1)
                dh1_dw2 = x[1] * deriv_sigmoid(sum_h1)
                dh1_db1 = deriv_sigmoid(sum_h1)

                # Neuron h2
                dh2_dw3 = x[0] * deriv_sigmoid(sum_h2)
                dh2_dw4 = x[1] * deriv_sigmoid(sum_h2)
                dh2_db2 = deriv_sigmoid(sum_h2)

                # Update weights and biases
                # Neuron 1
                self.h1.weights[0] -= learn_rate * dL_dypred * dypred_dh1 * dh1_dw1
                self.h1.weights[1] -= learn_rate * dL_dypred * dypred_dh1 * dh1_dw2
                self.h1.bias -= learn_rate * dL_dypred * dypred_dh1 * dh1_db1

                # Neuron 2
                self.h2.weights[0] -= learn_rate * dL_dypred * dypred_dh2 * dh2_dw3
                self.h2.weights[1] -= learn_rate * dL_dypred * dypred_dh2 * dh2_dw4
                self.h2.bias -= learn_rate * dL_dypred * dypred_dh2 * dh2_db2

                # Neuron o1
                self.o1.weights[0] -= learn_rate * dL_dypred * dypred_dw5
                self.o1.weights[1] -= learn_rate * dL_dypred * dypred_dw6
                self.o1.bias -= learn_rate * dL_dypred * dypred_db3

            # Calculate total loss at the end of each epoch
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(y_trues, y_preds)
                print("Epoch %d loss: %.6f" % (epoch, loss))


# Define dataset
data = np.array([
    [-2, -1],  # Alice 133lb - 135, 65in - 66
    [25, 6],   # Bob 160, 72
    [17, 4],   # Charlie 152, 70
    [-15, -6], # Diana 120, 60
])
y_trues = np.array([
    0, # Alice
    1, # Bob
    1, # Charlie
    0, # Diana
])

# Train our neural network!
network = NeuralNetwork()
network.train(data, y_trues)

# Make some predictions
emily = np.array([-7, -3]) # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
print("Emily: %.3f" % network.feedforward(emily)) # 0.951 - F
print("Frank: %.3f" % network.feedforward(frank)) # 0.039 - M
