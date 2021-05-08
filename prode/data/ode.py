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

        self.ode_func   = ode_func
        self.ode_order  = ode_order
        self.ode_params = ode_params

        self.grad_fn    = gradients.get_grad_fn(self.ode_order)


    def residual_loss(self, loss):
        '''Residual loss function definition.'''

        @tf.function
        def res_loss_fn(model, inputs):

            if isinstance(inputs, (list, tuple)):
                ic, t = inputs[0], inputs[1]
                input_is_list = True
            else:
                ic, t = tf.split(inputs, [inputs.shape[1]-1, 1], axis=1)
                input_is_list = False

            grads    = self.grad_fn(input_is_list, model, ic, t)

            res_pred = self.ode_func(ic, t, grads, self.ode_params)
            res_true = tf.zeros(res_pred.shape)

            return grads[0], loss(res_true, res_pred)
        
        return res_loss_fn
