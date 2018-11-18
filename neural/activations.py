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
