import numpy as np
import tensorflow as tf

from .adapter import LossWeightsAdapter


@tf.keras.utils.register_keras_serializable(package='PrODE', name='NTKAdapter')
class NTKAdapter(LossWeightsAdapter):
    """Adaptive loss weights technique for PINNs based on Neural Tangent Kernel
    (NTK) theory.

    See:
        Sifan Wang, Xinling Yu, Paris Perdikaris
        `When and why PINNs fail to train: A neural tangent kernel perspective`
        https://arxiv.org/abs/2007.14527"""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        data_ids,
        weights=None,
        freq=1,
        name='ntk_adapter'):
        super(NTKAdapter, self).__init__(
            data_ids=data_ids,
            weights=weights,
            freq=freq,
            name=name)


    # Configuration update
    ###########################################################################
    def get_config(self):
        config = super().get_config().copy()
        config.update({'alpha': self.alpha})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    # Calling
    ###########################################################################
    def __call__(self, model, losses, tape=None):

        if self.iter % self.freq == 0:
            # TODO
        self.iter += 1
