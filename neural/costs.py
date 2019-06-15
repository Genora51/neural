# -*- coding: utf-8 -*-
import numpy as np


# Cost functions
def quad_cost(out, target):
    return np.square(target - out) / 2
quad_cost.derivative = lambda out, target: out - target


def cross_entropy(out, target):
    return -target * np.log(out)
cross_entropy.derivative = lambda out, target: -np.true_divide(target, out)
