import tensorflow as tf

from .dataset import DataSet
from . import gradients


class ODE(DataSet):
    '''ODE solver.'''

    def __init__(
        self,
        ode_func=None,
        ode_order=0,
        ode_params=None,
        x_train=None,
        y_train=None,
        sample_weight=None,
        x_valid=None,
        y_valid=None,
        x_test=None,
        y_test=None,
        fname_train=None,
        fname_valid=None,
        fname_test=None,
        col_x=None,
        col_y=None,
        standardize=False):
        super(ODE, self).__init__(
            x_train=x_train,
            y_train=y_train,
            x_valid=x_valid,
            y_valid=y_valid,
            x_test=x_test,
            y_test=y_test,
            fname_train=fname_train,
            fname_valid=fname_valid,
            fname_test=fname_test,
            col_x=col_x,
            col_y=col_y,
            standardize=standardize)

        if isinstance(x_train, (list, tuple)):
            self.input_is_list = True
        else:
            self.input_is_list = False

        self.ode_func   = ode_func
        self.ode_params = ode_params

        self.grad_fn    = gradients.get_grad_fn(ode_order)


    def residual_loss(self, loss):
        '''Residual loss function definition.'''

        def res_loss_fn(model, inputs, training=True):

            if self.input_is_list:
                ic, t = inputs[0], inputs[1]
            else:
                ic, t = tf.split(inputs, [inputs.shape[1]-1, 1], axis=1)

            # Evaluate gradients
            grads    = self.grad_fn(self.input_is_list, model, ic, t, training)
            y_pred   = grads[0]     # Zero-order gradient

            # Evaluate residual
            res_pred = self.ode_func(ic, t, grads, self.ode_params)
            res_true = tf.zeros_like(res_pred)
            res_loss = loss(res_true, res_pred)

            return y_pred, res_loss
        
        return res_loss_fn
