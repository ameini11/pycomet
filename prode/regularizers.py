import tensorflow as tf
import tensorflow.keras.regularizers as reg



def get(name="l1+l2", l1=0., l2=0., weights_SP=None):
    return (
        reg.L1(l1=l1)
        if name == "l1"
        else reg.L2(l2=l2)
        if name == "l2"
        else reg.L1L2(l1=l1, l2=l2)
        if name == "l1+l2"
        else FineTuningRegularizer(l1=l1, l2=l2, weights_SP=weights_SP)
        if name == "fine_tuning"
        else None
    )


@tf.keras.utils.register_keras_serializable(package='PrODE', name='FineTuningRegularizer')
class FineTuningRegularizer(reg.L1L2):
    """Transfer learning technique for fine-tuning.
    See:
        Xuhong Li, Yves Grandvalet, Franck Davoine
        `A baseline regularization scheme for transfer learning with convolutional neural networks`
        https://doi.org/10.1016/j.patcog.2019.107049"""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        l1=0.,
        l2=0.,
        weights_SP=None,
        name='fine_tuning_reg'):
        super(FineTuningRegularizer, self).__init__(l1=l1, l2=l2)

        self.name = name

        # List of weights of pre-trained model: Starting Point in fine-tuning
                # inp.regularization.update({'weights_SP': model.net.trainable_weights})

        self.weights_SP = weights_SP
        if weights_SP is not None:
            self.weights_SP_names = [w.name for w in weights_SP]
        else:
            print("\n[PrODE]: Warning! `FineTuningRegularizer` is not given "
                  "any Starting Point weights `weights_SP`.")


    # Configuration update
    ###########################################################################
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'name':             self.name,
            'weights_SP':       self.weights_SP,
            'weights_SP_names': self.weights_SP_names
            })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


    # Calling
    ###########################################################################
    def __call__(self, x):
        if self.weights_SP is not None and x.name in self.weights_SP_names:
            idx = self.weights_SP_names.index(x.name)
            x = x - self.weights_SP[idx]
        return super(FineTuningRegularizer, self).__call__(x)
