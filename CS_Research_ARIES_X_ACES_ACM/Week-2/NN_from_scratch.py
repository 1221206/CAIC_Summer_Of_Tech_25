import numpy as np
# np.random.seed() for reproducablity
# assume all layers are nx1
# architecture for activation Sigmoid->ReLU->ReLU->......->softmax
# needs param dict, caches: linear cache(A,W,b), activation cache, hence use OOPS!!!
"""How do we want to call NN
    1. give layers list
    2. Form W,B
    3.Assign Activation function from architecture
"""

def ReLU(Z):
    return np.maximum(Z,0)  # elementsie 

def Softmax(X):
    exp_shifted = np.exp(X - np.max(X, axis=0, keepdims=True))
    return exp_shifted / np.sum(exp_shifted, axis=0, keepdims=True)

def sigmoid(X):
    return 1/(1+np.exp(-X))

def grad_ReLU(Z):
    return (Z > 0).astype(float)

def grad_sigmoid(Z):
    t=sigmoid(Z)
    return t*(1-t)

def grad_softmax():

    return

def to_one_hot(Y):
        one_hot_Y = np.zeros((Y.size,Y.max()+1))
        one_hot_Y[np.arange(Y.size),Y] = 1
        return one_hot_Y.T
    

class Layer:
    def __init__(self, input_dim, output_dim, activation):
        self.W = np.random.randn(output_dim, input_dim) * 0.01
        self.b = np.zeros((output_dim, 1))
        self.activation = activation
        self.cache = {}

    def forward(self, A_prev):
        Z = np.dot(self.W, A_prev) + self.b
        self.cache['A_prev'], self.cache['Z'] = A_prev, Z

        if self.activation == 'relu':
            A = ReLU(Z)
        elif self.activation == 'sigmoid':
            A = sigmoid(Z)
        elif self.activation == 'softmax':
            A = Softmax(Z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        
        self.cache['A'] = A
        return A,Z

    def backward(self, dA, activation_deriv):
        A_prev, Z = self.cache['A_prev'], self.cache['Z']
        
        if activation_deriv == 'relu':
            dZ = dA * grad_ReLU(Z)
        elif activation_deriv == 'sigmoid':
            dZ = dA * grad_sigmoid(Z)
        elif activation_deriv== 'softmax':
            dZ = dA
        else:
            raise ValueError(f"Unsupported derivative: {activation_deriv}")

        m = A_prev.shape[1]
        dW = (1 / m) * np.dot(dZ, A_prev.T)       
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.W.T, dZ)

        return dA_prev, dW, db

# -------- Neural Network Model -------- #
class NeuralNetwork:
    def __init__(self, layers_dims, activations):
        assert len(layers_dims) - 1 == len(activations)
        self.layers = []
        self.parameters={}
        self.caches={}
        for i in range(len(activations)):
            layer=Layer(layers_dims[i], layers_dims[i+1], activations[i])
            self.layers.append(layer)
            self.parameters[f"W{i}"]=layer.W
            self.parameters[f"b{i}"]=layer.b
    def forward(self, X):
        A = X
        self.caches["A0"]=X
        for i, layer in enumerate(self.layers):
            A_prev=A
            A,Z= layer.forward(A_prev)
            self.caches[f"Z{i+1}"] = Z
            self.caches[f"A{i+1}"] = A
        return A

    def backward(self, Y_hat, Y):
        grads = []
        m = Y.shape[0]
        one_hot_Y= to_one_hot(Y)
        dA = Y_hat - one_hot_Y  # derivative of cross-entropy loss with softmax

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            activation_deriv = layer.activation if layer.activation != 'softmax' else 'softmax'  # handled differently
            dA, dW, db = layer.backward(dA, activation_deriv)
            grads.insert(0, (dW, db))

        return grads

    def update_params(self, grads, learning_rate):
        for i, layer in enumerate(self.layers):
            dW, db = grads[i]
            layer.W -= learning_rate * dW
            layer.b -= learning_rate * db

    def compute_loss(self, Y_hat, Y):
        m = Y.shape[0]
        Y_one_hot= to_one_hot(Y)
        loss = -np.sum(Y_one_hot * np.log(Y_hat + 1e-8)) / m  # cross-entropy loss
        return loss

