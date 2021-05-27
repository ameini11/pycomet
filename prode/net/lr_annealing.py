import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='PrODE', name='LearningRateAnnealing')
class LearningRateAnnealing(object):
    """Learning rate annealing for physics-informed neural networks.
    See:
        Sifan Wang, Yujun Teng, Paris Perdikaris
        `Understanding and mitigating gradient pathologies in physics-informed neural networks`
        https://arxiv.org/abs/2001.04536"""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        loss_weights,
        alpha=0.9,
        name='lr_annealing'):

        self.name  = name
        self.alpha = tf.clip_by_value(alpha, 0., 1.)
        self.loss_weights = tf.convert_to_tensor(loss_weights)


    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
                    'name':         self.name,
                    'alpha':        self.alpha,
                    'loss_weights': self.loss_weights
                    }
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    # Calling
    ###########################################################################
    def __call__(self, model, res_loss, losses, tape):

        # Get the maximum gradient value from residual loss
        grad_res     = tape.gradient(res_loss, model.trainable_vars)
        grad_res_max = tf.reduce_max(tf.math.abs(grad_res))

        # Get the mean of the gradient magnitudes for each remaining loss term
        grads        = [ tape.gradient(loss, model.trainable_vars) for loss in losses ]
        grads_mean   = tf.convert_to_tensor([ tf.reduce_mean(tf.math.abs(grad)) for grad in grads ])

        # Update the loss weights
        loss_weights_hat = tf.math.divide(grad_res_max, grads_mean)
        self.loss_weights = (1. - self.alpha) * self.loss_weights + self.alpha * loss_weights_hat
