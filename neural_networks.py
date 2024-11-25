import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import logging


result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# activation functions and their derivatives
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr # learning rate
        self.activation_fn = activation # activation function
        
        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

        # Store activations and gradients
        self.hidden_activation = None
        self.output_activation = None
        self.dW1, self.db1, self.dW2, self.db2 = None, None, None, None

    def forward(self, X):
        # Apply activation to hidden layer
        z1 = X.dot(self.W1) + self.b1
        if self.activation_fn == 'tanh':
            self.hidden_activation = tanh(z1)
        elif self.activation_fn == 'relu':
            self.hidden_activation = relu(z1)
        elif self.activation_fn == 'sigmoid':
            self.hidden_activation = sigmoid(z1)

        if self.hidden_activation.shape[1] != 3:
            raise ValueError(
                f"Hidden activations must have 3 neurons for 3D visualization. Current shape: {self.hidden_activation.shape}"
            )

        # Compute output layer (hidden to output)
        z2 = self.hidden_activation.dot(self.W2) + self.b2
        self.output_activation = sigmoid(z2)  # Sigmoid for binary classification
        return self.output_activation

    def backward(self, X, y):
        # Output layer gradient
        output_error = self.output_activation - y
        self.dW2 = self.hidden_activation.T.dot(output_error)
        self.db2 = np.sum(output_error, axis=0, keepdims=True)

        # Hidden layer gradient
        if self.activation_fn == 'tanh':
            hidden_error = output_error.dot(self.W2.T) * tanh_derivative(self.hidden_activation)
        elif self.activation_fn == 'relu':
            hidden_error = output_error.dot(self.W2.T) * relu_derivative(self.hidden_activation)
        elif self.activation_fn == 'sigmoid':
            hidden_error = output_error.dot(self.W2.T) * sigmoid_derivative(self.hidden_activation)

        self.dW1 = X.T.dot(hidden_error)
        self.db1 = np.sum(hidden_error, axis=0, keepdims=True)

        # Update weights
        self.W1 -= self.lr * self.dW1
        self.b1 -= self.lr * self.db1
        self.W2 -= self.lr * self.dW2
        self.b2 -= self.lr * self.db2


def generate_data(n_samples=100):
    np.random.seed(0)
    # Generate input
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    # training steps, calling forward and backward function
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)
    
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()
        
    # hidden space
    hidden_features = mlp.hidden_activation
    x_range = np.linspace(-1.5, 1.5, 10)
    y_range = np.linspace(-1.5, 1.5, 10)
    xx, yy = np.meshgrid(x_range, y_range)

    for i in range(mlp.W1.shape[1]):  # Iterate over each neuron
        # Extract weights and bias for the i-th hidden neuron
        w1, w2 = mlp.W1[:, i]
        b = mlp.b1[0, i]

        # Compute z-coordinates for the plane
        z = (-w1 * xx - w2 * yy - b) / (1e-5 + mlp.W2[i, 0])  # Avoid division by zero

        # Add the plane to the plot
        ax_hidden.plot_surface(xx, yy, z, alpha=0.2, color='tan', rstride=1, cstride=1)

    # Plot hidden space points
    ax_hidden.scatter(
        hidden_features[:, 0],
        hidden_features[:, 1],
        hidden_features[:, 2],
        c=y.ravel(),
        cmap='bwr',
        alpha=0.7
    )
    ax_hidden.set_title(f"Hidden Space at Step {frame * 10}")

    # Input space decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = mlp.forward(grid).reshape(xx.shape) 
    ax_input.contourf(xx, yy, preds, levels=50, cmap='coolwarm', alpha=0.6)
    ax_input.contour(xx, yy, preds, levels=[0.5], colors='black', linewidths=1.5) 
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolors='k')
    ax_input.set_title(f"Input Space at Step {frame * 10}")

    # gradients
    # Define positions for nodes in the network (input layer, hidden layer, output layer)
    input_layer_pos = [(0.1, 0.8), (0.1, 0.6)]  # Positions for x1, x2
    hidden_layer_pos = [(0.5, 0.9), (0.5, 0.7), (0.5, 0.5)]  # Positions for h1, h2, h3
    output_layer_pos = [(0.9, 0.7)]  # Position for y

    input_labels = ['x1', 'x2']
    hidden_labels = ['h1', 'h2', 'h3']
    output_labels = ['y']

    # Plot input layer nodes with labels
    for pos, label in zip(input_layer_pos, input_labels):
        ax_gradient.scatter(*pos, s=500, c='blue')  # Input nodes
        ax_gradient.text(pos[0] - 0.05, pos[1], label, fontsize=12, ha='right', va='center')  # Label nodes

    # Define positions for nodes in the network (input layer, hidden layer, output layer)
    input_layer_pos = [(0.1, 0.8), (0.1, 0.6)]  # Positions for x1, x2
    hidden_layer_pos = [(0.5, 0.9), (0.5, 0.7), (0.5, 0.5)]  # Positions for h1, h2, h3
    output_layer_pos = [(0.9, 0.7)]  # Position for y

    input_labels = ['x1', 'x2']
    hidden_labels = ['h1', 'h2', 'h3']
    output_labels = ['y']

    # Plot input layer nodes with labels
    for pos, label in zip(input_layer_pos, input_labels):
        ax_gradient.scatter(*pos, s=500, c='blue')  # Input nodes
        ax_gradient.text(pos[0] - 0.05, pos[1], label, fontsize=12, ha='right', va='center')  # Label nodes

    # Plot hidden layer nodes with labels
    for pos, label in zip(hidden_layer_pos, hidden_labels):
        ax_gradient.scatter(*pos, s=500, c='blue')  # Hidden nodes
        ax_gradient.text(pos[0] + 0.05, pos[1], label, fontsize=12, ha='left', va='center')  # Label nodes

    # Plot output layer nodes with labels
    for pos, label in zip(output_layer_pos, output_labels):
        ax_gradient.scatter(*pos, s=500, c='blue')  # Output node
        ax_gradient.text(pos[0] + 0.05, pos[1], label, fontsize=12, ha='left', va='center')  # Label nodes

    # Define maximum line thickness
    max_linewidth = 5.0  # Maximum allowable line thickness

    # Draw connections and scale edge thickness by gradient magnitude
    for i, input_pos in enumerate(input_layer_pos):
        for j, hidden_pos in enumerate(hidden_layer_pos):
            weight_gradient = np.abs(mlp.dW1[i, j])  # Gradient magnitude for input-to-hidden weights
            linewidth = min(max_linewidth, weight_gradient * 5)  # Cap line thickness at max_linewidth
            ax_gradient.plot(
                [input_pos[0], hidden_pos[0]], [input_pos[1], hidden_pos[1]],
                linewidth=linewidth, c='purple', alpha=0.6  # Scaled line width
            )

    for j, hidden_pos in enumerate(hidden_layer_pos):
        for k, output_pos in enumerate(output_layer_pos):
            weight_gradient = np.abs(mlp.dW2[j, k])  # Gradient magnitude for hidden-to-output weights
            linewidth = min(max_linewidth, weight_gradient * 5)  # Cap line thickness at max_linewidth
            ax_gradient.plot(
                [hidden_pos[0], output_pos[0]], [hidden_pos[1], output_pos[1]],
                linewidth=linewidth, c='purple', alpha=0.6  # Scaled line width
            )

    ax_gradient.set_title(f"Gradients at Step {frame * 10}")
    ax_gradient.axis('off')  # Remove axes for a cleaner look


import logging

def visualize(activation, lr, step_num):
    try:
        logging.debug(f"Starting visualization with Activation: {activation}, Learning Rate: {lr}, Steps: {step_num}")
        X, y = generate_data()
        logging.debug(f"Generated data - X shape: {X.shape}, y shape: {y.shape}")

        mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)
        logging.debug("Initialized MLP model")

        matplotlib.use('agg')
        fig = plt.figure(figsize=(21, 7))
        ax_hidden = fig.add_subplot(131, projection='3d')
        ax_input = fig.add_subplot(132)
        ax_gradient = fig.add_subplot(133)

        ani = FuncAnimation(
            fig,
            partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y),
            frames=step_num // 10,
            repeat=False
        )
        logging.debug("Animation created")

        result_dir = "results"
        os.makedirs(result_dir, exist_ok=True)
        gif_path = os.path.join(result_dir, "visualize.gif")

        ani.save(gif_path, writer='pillow', fps=10)
        logging.debug(f"Visualization saved to {gif_path}")
        plt.close()

    except Exception as e:
        logging.error(f"Error in visualize function: {str(e)}")
        raise

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)