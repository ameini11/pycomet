import tensorflow as tf

from pycomet import utils
from tensorflow_addons.image.interpolate_spline import _solve_interpolation
from tensorflow_addons.image.interpolate_spline import _apply_interpolation


class InterpolateSpline(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        points,
        values,
        order=2,
        regularization_weight=0.,
        scale="lin",
        *args,
        **kwargs
    ):
        self.order = order
        self.regularization_weight = regularization_weight
        # Scaling
        self.scale = {
            "lin": [lambda x: x, lambda x: x],
            "log": [lambda x: tf.math.log(x), lambda x: tf.math.exp(x)]
        }[scale]
        # Fit the spline to the observed data
        self.solve(points, values)

    # Methods
    ###########################################################################
    def solve(self, points, values):
        self.train_points = self._reshape_tensor(points, dim=3)
        self.train_values = self._reshape_tensor(values, dim=3)
        self.rbf = _solve_interpolation(
            train_points=self.train_points,
            train_values=self.scale[0](self.train_values),
            order=self.order,
            regularization_weight=self.regularization_weight
        )

    def apply(self, points, dim_out=3):
        query_points = self._reshape_tensor(points, dim=3)
        query_values = _apply_interpolation(
            query_points=query_points,
            train_points=self.train_points,
            w=self.rbf[0],
            v=self.rbf[1],
            order=self.order
        )
        query_values = self.scale[1](query_values)
        query_values = self._reshape_tensor(query_values, dim=dim_out)
        return query_values

    def grad(self, points, indices=0, order=0):
        raise NotImplementedError

    def _reshape_tensor(self, array, dim=3):
        tensor = tf.convert_to_tensor(array)
        shape = tuple(tensor.shape)
        if 1 <= dim <= 3:
            if int(dim - len(shape)) == 2:
                for i in (0,-1):
                    tensor = tf.expand_dims(tensor, i)
                return tensor
            elif int(dim - len(shape)) == 1:
                return tf.expand_dims(tensor, 0)
            elif int(dim - len(shape)) == 0:
                return tensor
            elif int(dim - len(shape)) == -1:
                return tf.squeeze(tensor, 0)
            elif int(dim - len(shape)) == -2:
                for i in (0,-1):
                    tensor = tf.squeeze(tensor, i)
                return tensor
        else:
            utils.raise_value_err(
                f'The input tensor in `{self.name}` is more than 3D.'
            )
