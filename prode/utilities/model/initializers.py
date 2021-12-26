import scipy
import tensorflow as tf
import tensorflow.keras.initializers as tf_init

from prode.utilities import utils


def get(identifier):
    identifiers = {
        "zeros": tf_init.Zeros(),
        "HeNormal": tf_init.HeNormal(),
        "HeUniform": tf_init.HeUniform(),
        "LecunNormal": tf_init.LecunNormal(),
        "LecunUniform": tf_init.LecunUniform(),
        "GlorotNormal": tf_init.GlorotNormal(),
        "GlorotUniform": tf_init.GlorotUniform(),
        "Orthogonal": tf_init.Orthogonal()
    }

    if isinstance(identifier, str):
        return identifiers[identifier]
    elif callable(identifier):
        if inspect.isclass(identifier):
            identifier = identifier()
        return identifier
    else:
        utils.raise_value_err(
            "Could not interpret initializer identifier: " + str(identifier)
        )
