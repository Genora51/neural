# -*- coding: utf-8 -*-
import numpy as np


# Cost functions
def quad_cost(out, target):
    return np.square(target - out) / 2
quad_cost.derivative = lambda out, target: out - target
# TODO: make cross entropy a thing
