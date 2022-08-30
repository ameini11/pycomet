import abc
import tensorflow as tf


class PhysicalModel(metaclass=abc.ABCMeta):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        *args,
        **kwargs
    ):
        pass

    # XDEs Residual
    ###########################################################################
    @abc.abstractmethod
    def xde(
        self,
        x,
        y,
        grads
    ):
        """Evaluates the residual error of an ODEs/PDEs system, given
        the inputs/outputs of the model and the `Gradient` object
        to compute derivatives.

        Args:
            x: Model inputs.
            y: Model outputs.
            grads: `Gradient` object to compute derivatives.

        Returns:
            A `tf.Tensor` (or list of `tf.Tensors`) with shape (batch_size, n),
            where 'n' is the number equations cosidered.
        """
        return tf.zeros(1)
