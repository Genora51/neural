# -*- coding: utf-8 -*-
import numpy as np


# Activation functions
# TODO: make softmax and other multi-var activations work
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
sigmoid.derivative = lambda x, y: y * (1 - y)


def tanh(x):
    return np.tanh(x)
tanh.derivative = lambda x, y: 1 - np.square(y)


def arctan(x):
    return np.arctan(x)
arctan.derivative = lambda x, y: 1 / (np.square(x) + 1)


def bent_id(x):
    return ((np.sqrt(np.square(x) + 1) - 1) / 2) + x
bent_id.derivative = lambda x, y: (x / (2 * np.sqrt(np.square(x) + 1))) + 1


def gaussian(x):
    return np.exp(-np.square(x))
gaussian.derivative = lambda x, y: -2 * x * y


def identity(x):
    return x
identity.derivative = lambda x, y: 1


def sinusoid(x):
    return np.sin(x)
sinusoid.derivative = lambda x, y: np.cos(x)


def softplus(x):
    return np.log(1 + np.exp(x))
softplus.derivative = lambda x, y: 1 / (1 + np.exp(-x))


def elu(x):
    return np.where(x < 0, np.exp(x) - 1, x)
elu.derivative = lambda x, y: np.where(x < 0, y + 1, 1)


def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)
leaky_relu.derivative = lambda x, y: np.where(x > 0, 1, 0.01)


def relu(x):
    return x * (x > 0)
relu.derivative = lambda x, y: 1 * (x > 0)


def sinc(x):
    return np.where(x == 0, 1, np.sin(x) / x)
sinc.derivative = lambda x, y: np.where(x == 0, 0, (np.cos(x) - y) / x)


def softsign(x):
    return x / (1 + np.abs(x))
softsign.derivative = lambda x, y: x / np.square(1 + np.abs(x))
