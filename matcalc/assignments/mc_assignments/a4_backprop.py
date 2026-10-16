"""
Assignment 4 -- Backpropagation: layers and a multilayer perceptron.

Textbook: Chapter 8.   Lab Manual: Part 4.

Conventions:
  * a mini-batch is a 2-D array with one example per ROW;
  * layer.forward(X) returns the output and stores what backward needs;
  * layer.backward(dY) receives dL/dY (shape of the output), stores the
    parameter gradients (layer.dW, layer.db) and returns dL/dX;
  * params() / grads() return lists of arrays in the same order.
Rules: NumPy only; no automatic differentiation.
"""
import numpy as np


class Linear:
    """Y = X W + b with W of shape (d_in, d_out) and b of shape (d_out,).

    Initialise W with rng.standard_normal((d_in, d_out)) * sqrt(1 / d_in)
    and b with zeros.
    """

    def __init__(self, d_in, d_out, rng):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def forward(self, X):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def backward(self, dY):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def params(self):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def grads(self):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError


class ReLU:
    """H = max(Z, 0).  Use the convention ReLU'(0) = 0."""

    def forward(self, Z):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def backward(self, dH):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def params(self):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def grads(self):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError


class Tanh:
    """H = tanh(Z)."""

    def forward(self, Z):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def backward(self, dH):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def params(self):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def grads(self):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError


class Sigmoid:
    """H = 1 / (1 + exp(-Z)), computed without overflow for |Z| up to 1e4."""

    def forward(self, Z):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def backward(self, dH):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def params(self):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def grads(self):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError


class MSELoss:
    """L = 1/(2N) ||Y - T||_F^2 for N rows.  forward(Y, T) -> float; backward() -> dL/dY."""

    def forward(self, Y, T):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def backward(self):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError


class SoftmaxCrossEntropy:
    """Mean cross-entropy of softmax(A) (row-wise) with integer labels y.

    forward(A, y) -> float (stable for logits of size 1e4); backward() -> dL/dA.
    """

    def forward(self, A, y):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def backward(self):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError


class MLP:
    """Linear -> act -> Linear -> act -> ... -> Linear, with softmax cross-entropy.

    sizes: e.g. [2, 64, 64, 3]; activation: "tanh" or "relu".
    Create the Linear layers in order with ONE rng = np.random.default_rng(seed).
    """

    def __init__(self, sizes, activation="tanh", seed=0):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def forward(self, X):
        """Return the logits."""
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def loss(self, X, y):
        """Forward pass plus loss; must be called before backward()."""
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def backward(self):
        """Backpropagate from the loss; fills every layer's grads and returns dL/dX."""
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def params(self):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError

    def grads(self):
        # TODO: implement this method, then delete the next line.
        raise NotImplementedError


def train(model, X, y, lr=0.2, epochs=2000, momentum=0.9):
    """Full-batch gradient descent with momentum, updating model.params() IN PLACE.

    v <- momentum * v - lr * grad;  p <- p + v.   Returns the list/array of
    losses, one per epoch (the loss before that epoch's update).
    """
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError


def accuracy(model, X, y):
    """Fraction of rows whose largest logit is at the true label."""
    # TODO: implement this function, then delete the next line.
    raise NotImplementedError
