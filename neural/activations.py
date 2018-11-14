# -*- coding: utf-8 -*-
import numpy as np


# FIXME: sigmoid is slightly broken, probably bias' fault
# Activation functions
def sigmoid(x):
    return np.reciprocal(1 + np.exp(-x))


def _sigmoid_p(x):
    # s = sigmoid(x)
    return np.multiply(x, (1 - x))
sigmoid.derivative = _sigmoid_p


def tanh(x):
    return np.tanh(x)
tanh.derivative = lambda x: 1 - np.square(x)
