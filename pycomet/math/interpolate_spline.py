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
        *args,
        **kwargs
    ):
        self.order = order
        self.regularization_weight = regularization_weight

        # Initialize training points
        self.train_points = self._reshape_tensor(points)
        self.train_values = self._reshape_tensor(values)

        # Fit the spline to the observed data
        self.w, self.v = _solve_interpolation(
            self.train_points,
            self.train_values,
            self.order,
            self.regularization_weight
        )

    # Methods
    ###########################################################################
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

    def apply(self, points):
        query_points = self._reshape_tensor(points)
        query_values = _apply_interpolation(
            query_points,
            self.train_points,
            self.w,
            self.v,
            self.order
        )
        return self._reshape_tensor(query_values, dim=2)

    def grad(self, points, indices=0, order=0):
        raise NotImplementedError
