import inspect
import tensorflow.python.keras.initializers as init

from pycomet import utils


def get(identifier):
  identifiers = {
    "Ones":          init.Ones(),
    "Zeros":         init.Zeros(),
    "HeNormal":      init.HeNormal(),
    "HeUniform":     init.HeUniform(),
    "LecunNormal":   init.LecunNormal(),
    "LecunUniform":  init.LecunUniform(),
    "GlorotNormal":  init.GlorotNormal(),
    "GlorotUniform": init.GlorotUniform(),
    "Orthogonal":    init.Orthogonal()
  }
  if (identifier is None):
    return identifiers["GlorotNormal"]
  elif isinstance(identifier, str):
    return identifiers[identifier]
  elif callable(identifier):
    if inspect.isclass(identifier):
      identifier = identifier()
    return identifier
  else:
    utils.raise_err(
      ValueError,
      f"Could not interpret initializer identifier: {str(identifier)}"
    )
