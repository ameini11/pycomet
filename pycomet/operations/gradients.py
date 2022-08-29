import numpy as np
import tensorflow as tf

from pycomet import utils


class Gradients(object):

    def __init__(self):
        self.grads = {}

    def __call__(self, y, x, i, j, order=1):
        key = (y.ref(), x.ref())
        if key not in self.grads:
            self.grads[key] = Gradient(y, x)
        return self.grads[key](i, j, order)

    def clear():
        self.grads = {}


class Gradient(object):

    def __init__(self, y, x):
        self.y, self.x = y, x
        self.dim_y, self.dim_x = tf.nest.map_structure(
            lambda x: x.shape[1], [self.y, self.x]
        )
        self.i, self.j = tf.nest.map_structure(
            lambda x: tuple(np.arange(x)), [self.dim_y, self.dim_x]
        )
        self.g = {}

    def __call__(self, i=[], j=[], order=1):
        i, j = tf.nest.map_structure(
            lambda x: tuple(tf.nest.flatten(x)), [i, j]
        )
        i = self.i if len(i) == 0 else i
        if any([not 0 <= k < self.dim_y for k in i]):
            utils.raise_value_err(
                f"Index value i={i} is not valid for output tensor."
            )
        j = self.j if len(j) == 0 else j
        if any([not 0 <= k < self.dim_x for k in j]):
            utils.raise_value_err(
                f"Index value j={j} is not valid for input tensor."
            )
        key = (i, order)
        if key not in self.g:
            if order == 0:
                self.g[key] = y
            else:
                y = tf.gather(self.y, i, axis=1)
                self.g[key] = self.eval_grad(y, self.x, order=order)
        return tf.gather(self.g[key], j, axis=1)

    def eval_grad(self, y, x, order):
        if order == 0:
            return y
        else:
            dydx = tf.gradients(y, x)[-1]
            return self.eval_grad(dydx, x, order=order-1)
