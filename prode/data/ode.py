import tensorflow as tf

from .dataset import DataSet


class ODE(DataSet):
    '''ODE solver.'''

    def __init__(
        self,
        ode,
        order=1,
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
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            y_test,
            fname_train,
            fname_valid,
            fname_test,
            col_x,
            col_y,
            standardize)

        self.ode   = ode
        self.order = order


    def residual_loss(self, loss):
        '''Residual loss function definition.'''

        @tf.function
        def res_loss_fn(model, inputs):

            if isinstance(inputs, (list, tuple)):
                ic, t = inputs[0], inputs[1]
            else:
                ic, t = tf.split(inputs, [inputs.shape[1]-1, 1], axis=1)

            with tf.GradientTape(persistent=True) as first:
                first.watch(t)
                with tf.GradientTape(persistent=True) as second:
                    second.watch(t)
                    x = [ic, t]
                    if not isinstance(inputs, (list, tuple)):
                        x = tf.stack(x, axis=1)
                    y = model(x, training=True)
                dy_dt = tf.stack([first.gradient(y_i, t) for y_i in \
                        tf.split(y, num_or_size_splits=y.shape[1], axis=1)]) \
                        if self.order > 0 else None
            d2y_dt2   = tf.stack([second.gradient(dy_i, t) for dy_i in \
                        tf.split(dy_dt, num_or_size_splits=dy_dt.shape[1], axis=1)]) \
                        if self.order > 1 else None

            res_pred = self.ode(ic, t, y, dy_dt, d2y_dt2)
            res_true = tf.zeros(res_pred.shape, dtype=tf.dtypes.float64)

            return y, loss(res_true, res_pred)
        
        return res_loss_fn
