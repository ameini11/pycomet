import tensorflow as tf
from tensorflow.python.keras import backend as K

from prode.net.weighted_loss.adapter import LossWeightsAdapter
from prode                           import utils


@tf.keras.utils.register_keras_serializable(package='PrODE', name='EmpiricalAdapter')
class EmpiricalAdapter(LossWeightsAdapter):
    """Adaptive loss weights technique for PINNs based on an empirical learning
    rate annealing algorithm.

    See:
        Sifan Wang, Yujun Teng, Paris Perdikaris
        `Understanding and mitigating gradient pathologies in physics-informed
        neural networks`
        https://arxiv.org/abs/2001.04536"""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        data_ids,
        weights=None,
        freq=1,
        alpha=0.8,
        name='empirical_adapter'):
        super(EmpiricalAdapter, self).__init__(
            data_ids=data_ids,
            weights=weights,
            freq=freq,
            name=name)

        self.alpha = tf.cast(tf.clip_by_value(alpha, 0., 1.), K.floatx())


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
    def __call__(self, model, tape=None, losses=None, *args, **kwargs):
        
        if self.iter % self.freq == 0:
            if 'res' not in losses:
                utils.warning("`losses` passed into `__call__` method of `{}` hasn't "
                        "any residual loss to be evaluated.".format(self.__name__))
                return

            # Get the maximum gradient value from residual loss
            grads_res = tape.gradient(losses['res'], model.trainable_variables)
            grads_res_max = self.max_grads(grads_res)

            # Get the mean of the gradient magnitudes for each remaining loss term
            for data_id, loss in losses.items():
                if data_id == 'res':
                    continue
                # Get the mean of the gradient magnitudes for each remaining loss term
                grads = tape.gradient(loss, model.trainable_variables)
                grads_mean = self.mean_grads(grads)

                # Update the loss weights
                self.weights[data_id] = self.alpha * grads_res_max / grads_mean + \
                    (1. - self.alpha) * self.weights[data_id]

        self.iter += 1


    # Util functions
    ###########################################################################
    def mean_grads(self, grads):
        mean = [ tf.reduce_mean(tf.math.abs(grad)) for grad in grads ]
        return tf.reduce_mean(mean)

    def max_grads(self, grads):
        m = [ tf.reduce_max(tf.math.abs(grad)) for grad in grads ]
        return tf.reduce_max(m)
