# *************************************************************************** #
#                                    PrODE                                    #
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
from prode.utilities.model import layers as pro_layers


# Architecture properties
# =============================================================================
architecture = 'NeqCtrlNN'

# TimeScaleNNs characterization -----------------------------------------------
names = ['NEQ3']

opnn_dim = {
    names[0]: {
        "inp": inp_gen.dim['inp'],
        "out": 3
    }
}

layer_size = {
    name: {
        "branch": [2+27, 36, 36, 12],
        "trunk":  [1, 36, 36, 12]
    } for i, name in enumerate(names)
}

activation = {
    name: {
        "branch": 'tanh',   # elu/gelu
        "trunk":  'sigmoid'
    } for name in names
}

kernel_initializer = 'GlorotNormal' #'GlorotNormal' 'zeros'

block = 'FFBlock' # 'FFBlock' 'ResPIBlock'
block_linear_out = False

opnn_end_scale = 'dense'

stacked_blocks = {
    "branch": True,
    "trunk": False
}

# Shifter
shifter = {
    name: {
        "layer_size": [inp_gen.dim['inp'][0], 16, 16, 16, 2],
        "activation": 'sigmoid',
        "stacked": False
    } for i, name in enumerate(names)
}

# Augmentation
augmented = False

# Regularizers ----------------------------------------------------------------
batch_norm = {
    "branch": None,
    "trunk":  None
}

regularization = {
    'name':         'l1+l2',                                      # 'fine_tuning'
    'l1':           1.e-3,
    'l2':           1.e-3,
    'weights_path': './saving/'+inp_gen.model_name+'_weights.h5'  # For `FineTuningRegularizer`
}

dropout_rate = 0

# I/O Transformations
# =============================================================================
# Global model ----------------------------------------------------------------
# *** Inputs
def branch_fn(x):
    T = inp_gen.input_space['Ttr']
    if T[1] != T[0]:
        T_norm = (x[:,0:1] - T[0]) / (T[1] - T[0])
    else:
        T_norm = x[:,0:1]/T[0]
    return tf.concat([T_norm, x[:,1:]], 1)

input_fn = {
    "branch": tf_layers.Lambda(branch_fn, name='branch_fn'),
    #"trunk": pro_layers.Scaling(slope=1.e7, name='trunk_fn', trainable=False)
    #"trunk": tf_layers.Lambda(lambda x: tf.math.reciprocal((x+1.e-15)*1.e7))
    "trunk": pro_layers.Log(
        base='e', slope=1.e7, offset=1.e-14, name='trunk_fn', trainable=False
    )
}

# OpNNs -----------------------------------------------------------------------
# opnn_out_trans = tf_layers.Lambda(tf.math.exp, name='output_fn')
# opnn_out_trans = tf_layers.Lambda(tf.math.sigmoid, name='output_fn')
opnn_out_trans = {
    name: [
        pro_layers.TrainableAct(
            activation='sigmoid',
            pos_output=True,
            name=f'output_fn_{name}_opnn_3'
        ) for _ in range(len(tf.nest.flatten(opnn_dim[name]["out"])))
    ] for name in names
}
