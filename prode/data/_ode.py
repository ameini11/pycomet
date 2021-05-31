import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split

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
        out_dim=1,
        time_idx=-1,
        train_data=None,
        valid_data=None,
        test_data=None,
        valid_split=0.,
        col_x=None,
        col_y=None,
        batch_size=32,
        valid_batch_size=32):
        super(ODE, self).__init__()

        self.out_dim = out_dim

        # Defining data ids
        self.data_ids = [
            'ics',  # Initial condition pts
            'pts',  # Anchor pts
            'res'   # Collocation pts (residual)
            ]

        # Training/Validation data
        self.valid_split = valid_split
        self.check_data_source(train_data)
        if valid_data:
            self.check_data_source(valid_data)
            train_dset = self.data_as_dict(train_data)
            valid_dset = self.data_as_dict(valid_data)
        elif valid_split > 0.:
            train_dset, valid_dset = get_train_valid(self.data_as_dict(train_data))
        else:
            train_dset = self.data_as_dict(train_data)
            valid_dset = None
        self.train = ODEDataSequence(train_dset, self.data_ids, batch_size=batch_size)
        if valid_dset:
            self.valid = ODEDataSequence(valid_dset, self.data_ids, batch_size=valid_batch_size)

        # Testing data
        if test_data:
            self.check_data_source(test_data)
            self.test = ODEDataSequence(self.data_as_dict(test_data), \
                self.data_ids)
        else:
            self.test = None

        self.input_is_list = True if self.train.num_input > 1 else False
        if time_idx not in [0,-1]:
            utils.raise_value_err(
                "`time_idx` passed into `{}` has to be 0 or -1."
                "It's the position (if list) or column (if array) of time "
                "input vector.".format(self.__name__))
        self.time_idx      = time_idx

        self.ode_func      = ode_func
        self.ode_params    = ode_params
        self.grad_fn       = gradients.get_grad_fn(ode_order)


    # Handling Data
    ###########################################################################
    def check_data_source(self, data):
        name = utils.get_var_name(data)
        if data is None:
            utils.raise_value_err("No `{}` passed into `{}`.".format(name, self.__name__))
        elif not isinstance(data, dict):
            utils.raise_value_err(
                "`{}` passed into `{}` has to be a dict:\n"
                " >> {} = {'data_source': 'array' or 'file',\n"
                "          'ics': initial condition pts (arrays or file path),\n"
                "          'pts': additional anchor pts (arrays or file path),\n"
                "          'res': collocation pts for residual (arrays or file path)}".format( \
                    name, self.__name__, name))

    def data_as_dict(self, data):
        data = list(map(lambda i: self.get_data(data, i), self.data_ids))
        return {self.data_ids[i]: data[i] for i in range(len(self.data_ids))}

    def get_data(self, data, data_id):
        name = utils.get_var_name(data)
        if data[data_id] is None or data['data_source'] is None:
            return None
        if data['data_source'] is 'array':
            # Data from lists of `numpy.array`
            if isinstance(data[data_id], (list,tuple)):
                return data[data_id]
            elif data_id == 'res':
                return data[data_id], np.empty([data[data_id].shape[0], self.out_dim])
            else:
                utils.raise_value_err("`{}[{}]` passed into `{}` has to be a list "
                    "or a tuple of arrays: [x, y]".format(name, data_id, self.__name__))
        elif data['data_source'] is 'file':
            # Data from `.csv` file
            if col_x is None or col_y is None:
                utils.raise_value_err("In `{}` you have to provide both "
                    "`col_x` and `col_y` to extract data from file.\nThey should "
                    "be the doublet 'start'-'end+1' columns (or a list of those) "
                    "of x and y data.".format(self.__name__))
            data_frame = pd.read_csv(data[data_id], sep=',', header=None).values
            if any(isinstance(i, (list,tuple)) for i in col_x):
                x_data = []
                for i in range(len(col_x)):
                    x_data.append(data_frame[:, col_x[i][0]:col_x[i][1]])
            else:
                x_data = data_frame[:,col_x[0]:col_x[1]]
            y_data = data_frame[:,col_y[0]:col_y[1]]
            if y_data.size == 0:
                y_data = np.empty(data[data_id].shape)
            return x_data, y_data
        else:
            utils.raise_value_err("`{}['data_source']` passed into `{}` "
                "has to be equal to 'array' or 'file'.".format(name, self.__name__))


    # Splitting Training/Validation Data
    ###########################################################################
    def get_train_valid(self, data):
        train, valid = {}, {}
        for id_i, data_i in data:
            if data_i == None:
                train[id_i], valid[id_i] = None, None
            else:
                train[id_i], valid[id_i] = self.split_data(*data_i)
        return train, valid

    def split_data(self, x, y):
        x_len = len(x) if isinstance(x, (list,tuple)) else 1
        data  = (*x, y) if x_len > 1 else (x, y)
        split = train_test_split(*data, test_size=self.valid_split)
        x, x_val = [], []
        for i in range(len(data)):
            if i < x_len:
                x.append(split[i*2])
                x_val.append(split[i*2+1])
            else:
                y, y_val = split[i*2], split[i*2+1]
        return (x, y), (x_val, y_val)


    # Handling Residual Loss
    ###########################################################################
    def residual_loss(self, loss):
        '''Residual loss function definition.'''

        def res_loss_fn(model, inputs, training=True):

            if self.input_is_list:
                ic, t = inputs[self.time_idx-1], inputs[self.time_idx]
            else:
                sizes = [inputs.shape[1]-1, 1]
                if self.time_idx == 0:
                    sizes.reverse()
                ic, t = tf.split(inputs, sizes, axis=1)

            # Evaluate gradients
            grads    = self.grad_fn(self.input_is_list, model, ic, t, training)

            # Evaluate residual
            res_pred = self.ode_func(ic, t, grads, self.ode_params)
            res_true = tf.zeros_like(res_pred)
            res_loss = loss(res_true, res_pred)

            return res_pred, res_loss
        
        return res_loss_fn
