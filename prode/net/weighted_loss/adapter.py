import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='PrODE', name='LossWeightsAdapter')
class LossWeightsAdapter(object):
    """Base class for adaptive loss weights techniques."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        data_ids,
        weights=None,
        freq=1,
        name='loss_weights_adapter'):

        if weights is None:
            self.weights = dict()
            for k in data_ids:
                self.weights[k] = 1.
        self.data_ids = data_ids

        self.freq = freq
        self.iter = 0

        self.name = name


    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
                    'weights':  self.weights,
                    'data_ids': self.data_ids,
                    'freq':     self.freq,
                    'iter':     self.iter,
                    'name':     self.name
                    }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    # Calling
    ###########################################################################
    def __call__(self, model, *args, **kwargs):
        pass
