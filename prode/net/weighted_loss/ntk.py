import numpy as np
import tensorflow as tf

from .adapter import LossWeightsAdapter
from .... import utils


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


    # Calling
    ###########################################################################
    def __call__(self, model, tape=None, pred=None, *args, **kwargs):
        if self.iter % self.freq == 0:
            ntk = self.eval_diag_ntk(model, pred, tape)
            self.update_weights(ntk)
        self.iter += 1

    def eval_diag_ntk(self, model, pred, tape):
        ntk = {}
        for data_id in self.data_ids:
            n, d = pred[data_id].shape
            _ntk = tf.zeros([n*d,])
            for i in range(n):
                for j in range(d):
                    grad = tape.gradient(pred[data_id][i][j], model.trainable_vars)
                    # grad = np.concatenate([np.abs(g).flatten() for g in grad])
                    _ntk[i*d+j] = np.dot(grad, grad)
            ntk[data_id] = _ntk
        return ntk

    def update_weights(self, ntk):
        normalization_grad = sum([ tf.math.reduce_sum(ntk_v) for ntk_v in ntk.values()])
        for data_id in self.data_ids:
            self.weights[data_id] = normalization_grad / tf.math.reduce_sum(ntk[data_id])
