# *************************************************************************** #
#                                    PyCOMET                                    #
# --------------------------------------------------------------------------- #
#                     Machine-Learning-Based Approximators                    #
#                  for Ordinary Differential Equations (ODEs)                 #
#                                                                             #
# *************************************************************************** #

# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining the Neural Network architecture
# --------------------------------------------------------------------------- #


import tensorflow as tf
import inputs.general as inp_gen

from tensorflow.python.keras import layers as tf_layers
from pycomet.utilities.model import layers as pcm_layers


# Architecture properties
# =============================================================================
architecture = 'OpNN'

end_scale = 'dense'

# DeepONet characterization ---------------------------------------------------
layer_size = {
    "branch": [inp_gen.dim['inp'][0], 16, 16, 16, 1],
    "trunk":  [inp_gen.dim['inp'][1], 16, 16, 16, 1]
}

activation = {
    "branch": 'sigmoid',
    "trunk": 'sigmoid'
}

kernel_initializer = 'GlorotNormal'

block = 'FFBlock' #'FFBlock' 'ResPIBlock'
block_linear_out = True

stacked_blocks = {
    "branch": False,
    "trunk": False
}

# Shifter
shifter = {
    "layer_size": [inp_gen.dim['inp'][0], 32, 32, 32, 2],
    "activation": 'sigmoid',
    "stacked": False,
    # "out_trans": tf_layers.Lambda(tf.math.softplus, name='shifter_out_fn')
    "out_trans": tf_layers.Lambda(
        lambda x: tf.math.log(tf.math.softplus(x)), name='shifter_out_fn'
    )
}

# Augmentation
augmented = False

# Regularizers ----------------------------------------------------------------
batch_norm = {
    "branch": None,
    "trunk":  None
}

regularization = {
    'name':         'l1+l2',                              # 'fine_tuning'
    'l1':           1.e-5,
    'l2':           1.e-5,
    'weights_path': './saving/'+inp_gen.model_name+'_weights.h5'  # For `FineTuningRegularizer`
}

dropout_rate = 0

# I/O Transformations
# =============================================================================
# Global model ----------------------------------------------------------------
# *** Inputs
input_fn = {
    # "branch": tf_layers.Lambda(lambda x: x, name='branch_fn'),
    "branch": pcm_layers.Log(
        base=10, slope=1., offset=0., name='branch_fn', trainable=False
    ),
    "trunk": tf_layers.Lambda(lambda x: x, name='trunk_fn')
    # "trunk": pcm_layers.Scaling(slope=1.e7, name='trunk_fn', trainable=False)
    # "trunk": pcm_layers.Log(
    #     base='e', slope=1.e7, offset=1.e-15, name='trunk_fn', trainable=False
    # )
}

output_fn = tf_layers.Lambda(tf.math.exp, name='output_fn')