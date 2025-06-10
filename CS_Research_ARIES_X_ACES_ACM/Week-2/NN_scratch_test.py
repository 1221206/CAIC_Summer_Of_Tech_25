#n= no of layers, m= dim each layer
# architecture for activation Sigmoid->ReLU->ReLU->......->softmax
"""How do we want to call NN
    1. give layers list
    2. Form W,B
    3.Assign Activation function from architecture
"""
import numpy as np
import matplotlib.pyplot as plt
from NN_from_scratch import NeuralNetwork

# Generate synthetic non-linear dataset (sinusoidal decision boundary)
np.random.seed(1)
m = 400
X = np.random.uniform(-1, 1, (2, m))
Y = (X[1] > np.sin(3 * X[0])).astype(int)  # Y = 1 if above sine wave, else 0

# Model architecture
layers_dims = [2, 32, 32, 16, 2]  # deeper & wider
  # Input, 2 hidden layers, output (2 classes)
activations = ['relu','relu', 'relu', 'softmax']
model = NeuralNetwork(layers_dims, activations)

# Training
epochs = 1000
lr = 0.1
losses = []

for epoch in range(epochs):
    Y_hat = model.forward(X)
    loss = model.compute_loss(Y_hat, Y)
    grads = model.backward(Y_hat, Y)
    model.update_params(grads, lr)
    losses.append(loss)

    if epoch % 100 == 0:
        preds = np.argmax(Y_hat, axis=0)
        acc = np.mean(preds == Y)
        print(f"Epoch {epoch} | Loss: {loss:.4f} | Accuracy: {acc:.2f}")

# ---------- Plotting decision boundary ---------- #
def plot_decision_boundary(model, X, Y):
    x_min, x_max = X[0].min() - 0.1, X[0].max() + 0.1
    y_min, y_max = X[1].min() - 0.1, X[1].max() + 0.1
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()].T
    Z = model.forward(grid)
    predictions = np.argmax(Z, axis=0).reshape(xx.shape)
    
    plt.contourf(xx, yy, predictions, cmap=plt.cm.Spectral, alpha=0.5)
    plt.scatter(X[0], X[1], c=Y, cmap=plt.cm.Spectral, edgecolors='k')
    plt.title("Decision Boundary")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

# Plot decision boundary
plot_decision_boundary(model, X, Y)

# Plot loss curve
plt.plot(losses)
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
