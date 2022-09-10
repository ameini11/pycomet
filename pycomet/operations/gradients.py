import numpy as np
import tensorflow as tf

from pycomet import utils


class Gradients(object):

    def __init__(self):
        self.grads = {}
        self.built = []

    def __call__(self, y, x, j=[], i=[], order=1, key=None):
        if key is None:
            utils.raise_value_err(
                'Set a key (one for each different call) to identify '
                'the specific gradient computation.'
            )
        if not any([isinstance(z, (tf.Tensor,np.ndarray)) for z in [y, x]]):
            utils.raise_value_err(
                "'y' and 'x' must be tf.Tensor/np.ndarray "
                "when computing gradients."
            )
        if key not in self.built:
            self.build(y, x, j, i, order, key)
        return self.grads[key](y, x)

    def build(self, y, x, j, i, order, key):
        if key not in self.grads:
            self.grads[key] = Gradient(y, x, j, i, order)
        self.built.append(key)


class Gradient(object):

    def __init__(self, y, x, j, i, order):
        # Tensor dimensions
        self.dim_y, self.dim_x = tf.nest.map_structure(
            lambda x: x.shape[1], [y, x]
        )
        # Gradient indices
        self.build_indices(j, i)
        # Gradient order
        self.order = order

    def __call__(self, y, x):
        if self.order == 0:
            dydx = y
        else:
            dydx = [
                self.compute_grad(
                    tf.gather(y, k, axis=1), x, order=self.order
                ) for k in self.j
            ]
            dydx = [tf.split(g, self.dim_x, axis=1) for g in dydx]
            dydx = [tf.concat(x, 1) for x in list(zip(*dydx))]
        dydx = [tf.gather(g, self.j, axis=1) for g in dydx]
        return utils.squeeze_list([dydx[k] for k in self.i])

    def build_indices(self, j, i):
        j, i = list(map(lambda x: tuple(tf.nest.flatten(x)), [j, i]))
        # Output indices
        self.j = tuple(np.arange(self.dim_y)) if len(j) == 0 else j
        if any([not 0 <= k < self.dim_y for k in self.j]):
            utils.raise_value_err(
                f"Index value j={self.j} is not valid for output tensor."
            )
        # Input indices
        self.i = tuple(np.arange(self.dim_x)) if len(i) == 0 else i
        if any([not 0 <= k < self.dim_x for k in self.i]):
            utils.raise_value_err(
                f"Index value i={self.i} is not valid for input tensor."
            )

    def compute_grad(self, y, x, order):
        if order == 0:
            return y
        else:
            dydx = tf.gradients(y, x)[-1]
            return self.compute_grad(dydx, x, order=order-1)
