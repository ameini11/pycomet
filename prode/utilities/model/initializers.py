import scipy
import tensorflow as tf
import tensorflow.keras.initializers as init

from prode.utilities import utils


def get(identifier):
    identifiers = {
        "zeros":              init.Zeros(),
        "HeNormal":           init.HeNormal(),
        "HeUniform":          init.HeUniform(),
        "LecunNormal":        init.LecunNormal(),
        "LecunUniform":       init.LecunUniform(),
        "GlorotNormal":       init.GlorotNormal(),
        "GlorotUniform":      init.GlorotUniform(),
        "Orthogonal":         init.Orthogonal()
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
