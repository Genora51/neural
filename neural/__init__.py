# -*- coding: utf-8 -*-

import numpy as np
from neural import activations, costs

__all__ = ["NeuralNetwork", "activations", "costs"]


class NeuralNetwork(object):
    """A Feed-forward Neural Network"""

    def __init__(self, sizes, activations, cost=costs.quad_cost):
        if not hasattr(activations, '__iter__'):
            activations = [activations] * (len(sizes) - 1)
        self.activations = activations
        self.cost = cost
        self.weights = []
        for i in range(1, len(sizes) - 1):
            r = 2 * np.random.random((sizes[i - 1] + 1, sizes[i] + 1)) - 1
            self.weights.append(r)
        r = 2 * np.random.random((sizes[-2] + 1, sizes[-1])) - 1
        self.weights.append(r)
        self.layer_count = len(self.weights)

    def train(self, x, y, alpha=0.2, epochs=100, batch_size=1, momentum=0):
        """Train the network on a set of input/output values."""
        # TODO: make biases work properly
        # Reshape target data
        prev_adj = [0] * self.layer_count
        ys = np.array(y)
        while ys.ndim < 2:
            ys = np.expand_dims(ys, -1)
        # Add bias to input data
        xs = np.array(x, ndmin=2)
        xs = np.concatenate((
            xs,
            np.ones((xs.shape[0], 1))
        ), axis=1)
        num_cases = xs.shape[0]
        # Chunk cases into batches
        if batch_size < 1:
            # Full-batch learning
            batch_size = ys.shape[0]
        i_batches = np.array_split(xs, num_cases // batch_size)
        o_batches = np.array_split(ys, num_cases // batch_size)
        # Iterate through batches each epoch
        for k in range(epochs):
            for ins, outs in zip(i_batches, o_batches):
                # Run NN (feed-forward step)
                layer_outputs = [ins]
                layer_inputs = []
                for l in range(self.layer_count):
                    dot_val = layer_outputs[l] @ self.weights[l]
                    activated = self.activations[l](dot_val)
                    layer_inputs.append(dot_val)
                    layer_outputs.append(activated)
                # Differentiate error with respect to output neurons
                error_deriv = self.cost.derivative(layer_outputs[-1], outs)
                # Differentiate neuron output with respect to neuron input
                deriv_out_net = self.activations[-1].derivative(
                    layer_inputs[-1], layer_outputs[-1]
                )
                # Chain rule: error with respect to neuron input
                deltas = [np.einsum('ij,ijk->ik', error_deriv, deriv_out_net)]
                for l in range(len(layer_outputs) - 2, 0, -1):
                    # More complicated for hidden layer (backprop step)
                    deriv_out_net = self.activations[l].derivative(
                        layer_inputs[l - 1], layer_outputs[l]
                    )
                    deriv_err_out = deltas[-1] @ self.weights[l].T
                    # Chain rule
                    delta = np.einsum(
                        'ij,ijk->ik',
                        deriv_err_out, deriv_out_net
                    )
                    deltas.append(delta)
                # Reverse to correct layer order
                deltas.reverse()
                # Update weights with backprop data
                for i in range(self.layer_count):
                    o_i = layer_outputs[i].T
                    d_j = deltas[i]
                    # Use previous momentum to adjust changes
                    dotted = (o_i @ d_j) + prev_adj[i]
                    # Update with learning rate (alpha)
                    self.weights[i] -= alpha * dotted
                    # Store previous changes for momentum adjustment
                    prev_adj[i] = dotted * momentum

    def __call__(self, x):
        """Run the network without training it."""
        inputs = np.array(x, ndmin=2)
        a = np.concatenate((
            inputs,
            np.ones((inputs.shape[0], 1))
        ), axis=1)
        for l in range(self.layer_count):
            a = self.activations[l](a @ self.weights[l])
        return np.squeeze(a)

    def test(self, x, y):
        """Compare inputs and outputs."""
        # FIXME: >1 output node
        out = self.__call__(x)
        predicted = np.array(y)
        error = self.cost(out, predicted)
        wrong = abs(np.round(out) - predicted)
        return wrong, error

    def save(self, filename):
        """Serialize a neural net's weights."""
        np.savez(filename, *self.weights)

    def load(self, filename):
        """Load a serialized net."""
        raw_weights = np.load(filename)
        self.weights = [raw_weights[x] for x in raw_weights]
