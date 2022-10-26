# -------------------------------- INPUT FILE ------------------------------- #
# Description:
# >> Defining custom objects for the NN architecture
# --------------------------------------------------------------------------- #


import tensorflow as tf

from keras import layers as tf_layers
from pycomet import config as cfg


# Custom objects
# =============================================================================
@tf.keras.utils.register_keras_serializable(package=cfg._NAME)
class OutputXform(tf_layers.Layer):
    # Initialization
    def __init__(self, **kwargs):
        super(OutputXform, self).__init__(**kwargs)
    # Calling
    def call(self, y, mask=None, training=None):
        y = tf.math.sigmoid(y)
        return tf.concat([1.-y, y],1)
