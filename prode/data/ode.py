import numpy as np
import tensorflow as tf

from .dataset import DataSet
from . import gradients
from .. import utils


class ODE(DataSet):
    '''ODE solver.'''

    def __init__(
        self,
        ode_func=None,
        ode_order=0,
        ode_params=None,
        x_train=None,
        *args,
        **kwargs):
        super(ODE, self).__init__(x_train=x_train, *args, **kwargs)

        if isinstance(x_train, (list, tuple)):
            self.input_is_list = True
        else:
            self.input_is_list = False

        self.ode_func   = ode_func
        self.ode_params = ode_params

        self.grad_fn    = gradients.get_grad_fn(ode_order)


    # def residual_loss(self, loss):
    #     '''Residual loss function definition.'''

    #     def res_loss_fn(model, inputs, training=True):

    #         if self.input_is_list:
    #             ic, t = inputs[0], inputs[1]
    #         else:
    #             ic, t = tf.split(inputs, [inputs.shape[1]-1, 1], axis=1)

    #         # Evaluate gradients
    #         grads    = self.grad_fn(self.input_is_list, model, ic, t, training)
    #         y_pred   = grads[0]     # Zero-order gradient

    #         # Evaluate residual
    #         res_pred = self.ode_func(ic, t, grads, self.ode_params)
    #         res_true = tf.zeros_like(res_pred)
    #         res_loss = loss(res_true, res_pred)

    #         return y_pred, res_loss
        
    #     return res_loss_fn


    def residual_loss(self, loss):
        '''Residual loss function definition.'''

        def res_loss_fn(model, inputs, training=True):

            if self.input_is_list:
                ic, t = inputs[0], inputs[1]
            else:
                ic, t = tf.split(inputs, [inputs.shape[1]-1, 1], axis=1)

            # Evaluate gradients
            grads_init = tf.TensorArray(tf.float64, size=0, dynamic_size=True, \
                            clear_after_read=False, tensor_array_name='grads')
            grads      = self.grad_fn(grads_init, self.input_is_list, model, ic, t, training)
            y_pred     = grads.read(0)     # Zero-order gradient

            # Evaluate residual
            res_pred = self.ode_func(ic, t, grads, self.ode_params)
            res_true = tf.zeros_like(res_pred)
            res_loss = loss(res_true, res_pred)

            return y_pred, res_loss
        
        return res_loss_fn





class ODE(Data):
    '''ODE solver.'''

    def __init__(
        self,
        ode_func=None,
        ode_order=0,
        ode_params=None,
        train_data=None,
        valid_data=None,
        test_data=None,
        col_x=None,
        col_y=None,
        batch_size=32):
        super(ODE, self).__init__()

        data_families = ['ics', 'pts', 'res']

        self.batch_size = {}
        if not isinstance(batch_size, dict):
            batch_size = 

        # Training data
        self.check_data_source(train_data)
        self.ics, self.pts, self.res = list(map( \
            lambda i: self.get_data(train_data, i), data_families))

        # Validation data
        self.check_data_source(valid_data)
        self.ics_val, self.pts_val, self.res_val = list(map( \
            lambda i: self.get_data(train_data, i), data_families))

        # Testing data
        self.check_data_source(test_data)
        self.ics_test, self.pts_test, self.res_test = list(map( \
            lambda i: self.get_data(train_data, i), data_families))

        self.input_is_list = True if isinstance(self.ics[0], (list, tuple)) else False
        self.ode_func      = ode_func
        self.ode_params    = ode_params
        self.grad_fn       = gradients.get_grad_fn(ode_order)


    def check_data_source(self, data):
        name = utils.get_var_name(data)
        if data is None:
            raise ValueError("[PrODE]: No `{}` passed into `{}`.".format(name, self.__name__))
        elif not isinstance(data, dict):
            raise ValueError(
                "\n[PrODE]: `{}` passed into `{}` has to be a dict:\n"
                "   >> 'data_source': 'array' or 'file';\n"
                "   >> 'ics': initial condition pts (array or file path);\n"
                "   >> 'pts': additional anchor pts (array or file path);\n"
                "   >> 'res': residual collocation pts (array or file path).".format(name, self.__name__))


    def get_data(self, data, data_family):
        name = utils.get_var_name(data)
        if data[data_family] is None or data['data_source'] is None:
            return [ np.array([]) ] * 2
        if data['data_source'] is 'array':
            # Data from lists of`numpy.array`
            if isinstance(data[data_family], (list,tuple)):
                return data[data_family]
            else:
                if data_family == 'res':
                    return data[data_family], np.empty([data[data_family].shape[0],0])
                else:
                    raise ValueError('[PrODE]: `{}[{}]` passed into `{}` has to '
                        'be a list or a tuple of arrays: [x, y]'.format( \
                            name, data_family, self.__name__))
        elif data['data_source'] is 'file':
            # Data from `.csv` file
            if col_x is None or col_y is None:
                raise ValueError('[PrODE]: In `{}` you have to provide both `col_x` '
                    'and `col_y` to extract data from file.'.format(self.__name__))
            data_frame = pd.read_csv(data[data_family], sep=',', header=None).values
            y_data = data_frame[:,col_y[0]:col_y[1]]
            if any(isinstance(i, list) for i in col_x):
                x_data = []
                for i in range(len(col_x)):
                    x_data.append(data_frame[:, col_x[i][0]:col_x[i][1]])
            else:
                x_data = data_frame[:,col_x[0]:col_x[1]], data_frame[:,col_y[0]:col_y[1]]
            return x_data, y_data
        else:
            raise ValueError("[PrODE]: `{}['data_source']` passed into `{}` "
                "has to be equal to 'array' or 'file'.".format(name, self.__name__))





    def get_batched_data(self, data, batch_size=None):

        dataset = tf.data.Dataset.from_tensor_slices(tuple(data))

        if self.online:
            self.train_x = self.geom.random_points(batch_size, "pseudo")
            self.train_y = self.func(self.train_x)
        elif self.train_x is None:
            if self.dist_train == "uniform":
                self.train_x = self.geom.uniform_points(self.num_train, True)
            else:
                self.train_x = self.geom.random_points(self.num_train, "sobol")
            self.train_y = self.func(self.train_x)
        return self.train_x, self.train_y

    # def residual_loss(self, loss):
    #     '''Residual loss function definition.'''

    #     def res_loss_fn(model, inputs, training=True):

    #         if self.input_is_list:
    #             ic, t = inputs[0], inputs[1]
    #         else:
    #             ic, t = tf.split(inputs, [inputs.shape[1]-1, 1], axis=1)

    #         # Evaluate gradients
    #         grads    = self.grad_fn(self.input_is_list, model, ic, t, training)
    #         y_pred   = grads[0]     # Zero-order gradient

    #         # Evaluate residual
    #         res_pred = self.ode_func(ic, t, grads, self.ode_params)
    #         res_true = tf.zeros_like(res_pred)
    #         res_loss = loss(res_true, res_pred)

    #         return y_pred, res_loss
        
    #     return res_loss_fn


    def residual_loss(self, loss):
        '''Residual loss function definition.'''

        def res_loss_fn(model, inputs, training=True):

            if self.input_is_list:
                ic, t = inputs[0], inputs[1]
            else:
                ic, t = tf.split(inputs, [inputs.shape[1]-1, 1], axis=1)

            # Evaluate gradients
            grads_init = tf.TensorArray(tf.float64, size=0, dynamic_size=True, \
                            clear_after_read=False, tensor_array_name='grads')
            grads      = self.grad_fn(grads_init, self.input_is_list, model, ic, t, training)
            y_pred     = grads.read(0)     # Zero-order gradient

            # Evaluate residual
            res_pred = self.ode_func(ic, t, grads, self.ode_params)
            res_true = tf.zeros_like(res_pred)
            res_loss = loss(res_true, res_pred)

            return y_pred, res_loss
        
        return res_loss_fn



