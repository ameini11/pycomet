import tensorflow as tf


class Data(object):
    """Data base class."""

    def __init__(self):
        self.train = None
        self.valid = None
        self.test  = None
        
        self.data_ids = None

    def get_dataset(self, x, y):
        if x is not None:
            dset_x = tf.data.Dataset.from_tensor_slices(tuple(x))
            dset_y = tf.data.Dataset.from_tensor_slices(tuple(y))
            return tf.data.Dataset.zip(tuple(x, y))
        else:
            None
