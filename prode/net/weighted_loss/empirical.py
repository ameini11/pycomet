import tensorflow as tf

from .adapter import LossWeightsAdapter


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

        self.alpha = tf.clip_by_value(alpha, 0., 1.)


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
                print("[PrODE]: Warning! `losses` passed into `__call__` method "
                    "of `{}` hasn't any residual loss to be evaluated.".format( \
                        self.__name__))
                return

            # Get the maximum gradient value from residual loss
            grad_res_max = tf.reduce_max(tf.math.abs(tape.gradient( \
                losses['res'], model.trainable_vars)))

            # Get the mean of the gradient magnitudes for each remaining loss term
            for data_id, loss in losses.items():
                if data_id == 'res':
                    continue
                # Get the mean of the gradient magnitudes for each remaining loss term
                grad_mean = tf.reduce_mean(tf.math.abs(tape.gradient(\
                    loss, model.trainable_vars)))

                # Update the loss weights
                self.weights[data_id] = self.alpha * grad_res_max / grads_mean + \
                    (1. - self.alpha) * self.weights[data_id]

        self.iter += 1
