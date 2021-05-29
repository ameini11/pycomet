import numpy as np
import tensorflow as tf

from .dataset import DataSet
from .generator import ODEDataSequence
from . import gradients
from .. import utils


class ODE(Data):
    '''ODE solver.'''

    # Initialization
    ###########################################################################
    def __init__(
        self,
        ode_func=None,
        ode_order=0,
        ode_params=None,
        train_data=None,
        valid_data=None,
        valid_split=0.,
        test_data=None,
        col_x=None,
        col_y=None,
        batch_size=32,
        valid_batch_size=32):
        super(ODE, self).__init__()

        self.data_ids = ['ics', 'pts', 'res']

        # Training data
        self.check_data_source(train_data)
        self.train = ODEDataSequence(self.data_as_dict(train_data), \
            self.data_ids, batch_size=batch_size)

        # Validation data
        if valid_data:
            self.check_data_source(valid_data)
            self.valid = ODEDataSequence(self.data_as_dict(valid_data), \
                self.data_ids, batch_size=valid_batch_size)
        elif valid_split > 0.:
            # TODO
            pass
        else:
            self.valid = None

        # Testing data
        if test_data:
            self.check_data_source(test_data)
            self.test = ODEDataSequence(self.data_as_dict(test_data), \
                self.data_ids)
        else:
            self.test = None

        self.input_is_list = True if self.train.num_input > 1 else False
        self.ode_func      = ode_func
        self.ode_params    = ode_params
        self.grad_fn       = gradients.get_grad_fn(ode_order)


    # Handling Data
    ###########################################################################
    def check_data_source(self, data):
        name = utils.get_var_name(data)
        if data is None:
            raise ValueError("[PrODE]: No `{}` passed into `{}`.".format(name, self.__name__))
        elif not isinstance(data, dict):
            raise ValueError(
                "[PrODE]: `{}` passed into `{}` has to be a dict:\n"
                "   >> 'data_source': 'array' or 'file';\n"
                "   >> 'ics': initial condition pts (arrays or file path);\n"
                "   >> 'pts': additional anchor pts (arrays or file path);\n"
                "   >> 'res': residual collocation pts (arrays or file path).".format(name, self.__name__))

    def data_as_dict(self, data):
        data = list(map(lambda i: self.get_data(data, i), self.data_ids))
        return {self.data_ids[i]: data[i] for i in range(len(self.data_ids))}

    def get_data(self, data, data_id):
        name = utils.get_var_name(data)
        if data[data_id] is None or data['data_source'] is None:
            return None
        if data['data_source'] is 'array':
            # Data from lists of`numpy.array`
            if isinstance(data[data_id], (list,tuple)):
                return data[data_id]
            elif data_id == 'res':
                return data[data_id], np.empty([data[data_id].shape[0],])
            else:
                raise ValueError('[PrODE]: `{}[{}]` passed into `{}` has to '
                    'be a list or a tuple of arrays: [x, y]'.format( \
                        name, data_id, self.__name__))
        elif data['data_source'] is 'file':
            # Data from `.csv` file
            if col_x is None or col_y is None:
                raise ValueError('[PrODE]: In `{}` you have to provide both `col_x` '
                    'and `col_y` to extract data from file.'.format(self.__name__))
            data_frame = pd.read_csv(data[data_id], sep=',', header=None).values
            if any(isinstance(i, (list,tuple)) for i in col_x):
                x_data = []
                for i in range(len(col_x)):
                    x_data.append(data_frame[:, col_x[i][0]:col_x[i][1]])
            else:
                x_data = data_frame[:,col_x[0]:col_x[1]], data_frame[:,col_y[0]:col_y[1]]
            y_data = data_frame[:,col_y[0]:col_y[1]]
            if y_data.size == 0:
                y_data = np.empty(data[data_id].shape)
            return x_data, y_data
        else:
            raise ValueError("[PrODE]: `{}['data_source']` passed into `{}` "
                "has to be equal to 'array' or 'file'.".format(name, self.__name__))


    # Handling Losses
    ###########################################################################
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
