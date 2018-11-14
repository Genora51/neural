# -*- coding: utf-8 -*-
import numpy as np


# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
sigmoid.derivative = lambda x: np.multiply(x, (1 - x))


def tanh(x):
    return np.tanh(x)
tanh.derivative = lambda x: 1 - np.square(x)
