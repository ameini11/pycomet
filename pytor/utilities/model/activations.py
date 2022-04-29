import tensorflow as tf
import tensorflow.python.keras.activations as act

from tensorflow.python.keras import backend as K
from prode.utilities import utils


def layer_wise_locally_adaptive(activation, n=1):
    """Layer-wise locally adaptive activation functions (L-LAAF).

    Examples:

    To define a L-LAAF ReLU with the scaling factor ``n = 10``:

    .. code-block:: python

        n = 10
        activation = f"LAAF-{n} relu"  # "LAAF-10 relu"

    References: `Jagtap et al., 2019 <https://arxiv.org/abs/1909.12228>`_.
    """
    a = tf.Variable(
        1./n, trainable=True, name='act_scaling', dtype=K.floatx()
    )
    return lambda x: activation(n*a*x)

def get(identifier):
    if identifier is None:
        return act.linear
    if isinstance(identifier, str):
        if identifier.startswith("LAAF"):
            identifier = identifier.split()
            n = float(identifier[0].split("-")[1])
            return layer_wise_locally_adaptive(get(identifier[1]), n=n)
        return {
            "abs":      tf.math.abs,
            "cos":      tf.math.cos,
            "elu":      act.elu,
            "exp":      act.exponential,
            "gelu":     act.gelu,
            "hard_sig": act.hard_sigmoid,
            "linear":   act.linear,
            "log":      tf.math.log,
            "relu":     act.relu,
            "selu":     act.selu,
            "sigmoid":  act.sigmoid,
            "sin":      tf.math.sin,
            "softmax":  act.softmax,
            "softplus": act.softplus,
            "softsign": act.softsign,
            "square":   tf.math.square,
            "swish":    act.swish,
            "tanh":     act.tanh,
        }[identifier]
    elif callable(identifier):
        return identifier
    else:
        utils.raise_value_err(
            f"Could not interpret activation function "
            f"identifier: {str(identifier)}"
        )
