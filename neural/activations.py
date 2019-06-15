# -*- coding: utf-8 -*-
import numpy as np


# Activation functions
class Activation:
    """Activation function decorator."""

    def __init__(self, derivative, onehot=True):
        self.onehot = onehot
        self.derivative = derivative

    def __call__(self, func):
        if self.onehot:
            def derivative(x, y):
                return np.apply_along_axis(
                    np.diag, -1,
                    self.derivative(x, y)
                )
        else:
            derivative = self.derivative
        func.derivative = derivative
        return func


@Activation(derivative=lambda x, y: y * (1 - y))
def sigmoid(x):
    s = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-s))


@Activation(derivative=lambda x, y: 1 - np.square(y))
def tanh(x):
    return np.tanh(x)


@Activation(derivative=lambda x, y: 1 / (np.square(x) + 1))
def arctan(x):
    return np.arctan(x)


@Activation(derivative=lambda x, y: (x / (2 * np.sqrt(np.square(x) + 1))) + 1)
def bent_id(x):
    return ((np.sqrt(np.square(x) + 1) - 1) / 2) + x


@Activation(derivative=lambda x, y: -2 * x * y)
def gaussian(x):
    return np.exp(-np.square(x))


@Activation(derivative=lambda x, y: 1)
def identity(x):
    return x


@Activation(derivative=lambda x, y: np.cos(x))
def sinusoid(x):
    return np.sin(x)


@Activation(derivative=lambda x, y: 1 / (1 + np.exp(-x)))
def softplus(x):
    return np.log(1 + np.exp(x))


@Activation(derivative=lambda x, y: np.where(x < 0, y + 1, 1))
def elu(x):
    return np.where(x < 0, np.exp(x) - 1, x)


@Activation(derivative=lambda x, y: np.where(x > 0, 1, 0.01))
def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)


@Activation(derivative=lambda x, y: 1 * (x > 0))
def relu(x):
    return x * (x > 0)


@Activation(derivative=lambda x, y: np.where(x == 0, 0, (np.cos(x) - y) / x))
def sinc(x):
    return np.where(x == 0, 1, np.sin(x) / x)


@Activation(derivative=lambda x, y: x / np.square(1 + np.abs(x)))
def softsign(x):
    return x / (1 + np.abs(x))


def _deriv_softmax(x, y):
    i = np.stack([np.eye(y.shape[-1])]*y.shape[0])
    mul = i - y[:, np.newaxis]
    return (mul.T * y.T).T


@Activation(derivative=_deriv_softmax, onehot=False)
def softmax(x):
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=-1)[:, None]
