import os
import sys
import numpy as np
import tensorflow as tf

from keras.layers import Concatenate
from keras.engine.functional import Functional
from tensorflow.python.keras import backend


# Inputs
nb_bins = 27
paths = {
    "pycomet": "/home/zanardi/Codes/ML/PyCOMET/pycomet",
    "neqopnn": ...,
    "cgopnn": ...,
    "saving": "./cgneq_deeponet/"
}

# Set dimensions
dim = {
    "inp": [nb_bins+2, 1, 1],
    "out": [1, nb_bins, 1]
}

# Load each models
neqopnn = tf.keras.models.load_model(paths["neqopnn"])
cgopnn = tf.keras.models.load_model(paths["cgopnn"])

# Preinput layer
sys.path.append(paths["pycomet"])
from pycomet.networks.chem.cg.adaptive.layers import CGThermoEncoder

# Function call
def call(x, dim):
    # Input manipulation
    x = tf.split(x, dim["inp"], axis=1)
    x = CGThermoEncoder(nb_bins=nb_bins)(x)
    y, tol = x[:-1], x[-1]
    # Kernel - Neq-DeepONet
    mask = neqopnn(x)
    # Kernel - CG-DeepONet
    y = tf.nest.flatten([y, mask])
    y = cgopnn(y)
    # Output manipulation
    y = Concatenate()(y)
    return y

# Input shape
n = neqopnn.inputs[-1].shape[0]
d = np.sum([i.shape[1] for i in neqopnn.inputs])
# Inputs/Outputs
inputs = tf.keras.Input(tensor=tf.zeros((n,d), dtype=backend.floatx()))
outputs = call(inputs, dim)
# Build model
net = Functional(inputs, outputs, name="CGNeqOpNN")

# Save model
from pycomet.model import _generate_api_inpfile
os.makedirs(paths["saving"], exist_ok=True)

net.save(
    paths["saving"] + '/SavedModel',
    overwrite=True,
    include_optimizer=False,
    save_format='tf',
    signatures=None,
    options=None,
    save_traces=True
)
_generate_api_inpfile(paths["saving"])

