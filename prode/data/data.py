import tensorflow as tf

from prode.utilities import utils


class Data(object):
    """Data base class."""


    # Initialization
    ###########################################################################
    def __init__(self, dim):

        self.name  = utils.get_class_name(self)

        self.dim   = dim

        self.train = None
        self.valid = None
        self.test  = None
        
        self.data_ids = None
        
        self.batch_size = None
        self.valid_batch_size = None
        self.validation_split = None


    # Util functions
    ###########################################################################
    def get_cardinality(self, x):
        if isinstance(x, (list,tuple)):
            return self.get_cardinality(x[0])
        else:
            return x.shape[0]
            
    def get_dataset(self, x, y, batch_size=None):
        if x is not None:
            if batch_size:
                return tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
            else:
                return tf.data.Dataset.from_tensor_slices((x, y))

