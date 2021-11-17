import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K

from varname import nameof
from prode.data.sequence import ODEDataSequence
from prode.data.data import Data
from prode.data import gradients
from prode.utilities import utils


class ODE(Data):
    '''ODE solver.'''

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        training=True,
        ode_func=None,
        ode_order=0,
        time_idx=-1,
        train_data=None,
        valid_data=None,
        test_data=None,
        validation_split=None,
        from_file=False,
        col_x=None,
        col_y=None,
        header=None,
        preproc_fn=None,
        batch_size=32,
        valid_batch_size=None
    ):
        super(ODE, self).__init__(dim=dim, training=training)

        # Defining data ids
        self.data_ids = tuple([
            'ics',  # Initial condition pts
            'pts',  # Anchor pts
            'res'   # Collocation pts (residual)
            ])
        
        # Only for data from file
        self.from_file = from_file
        self.col_x = col_x
        self.col_y = col_y
        self.header = header
        self.preproc_fn = preproc_fn

        # Training/Validation data
        if training:
            self.check_data_source(train_data, nameof(train_data))
            if valid_data:
                self.check_data_source(valid_data, nameof(valid_data))
                train_dset = self.data_as_dict(train_data)
                valid_dset = self.data_as_dict(valid_data)
            elif validation_split:
                self.validation_split = validation_split
                train_dset, valid_dset = self.get_train_valid(
                    self.data_as_dict(train_data)
                )
            else:
                train_dset = self.data_as_dict(train_data)
                valid_dset = None

            self.train = ODEDataSequence(train_dset, self.data_ids, \
                batch_size=batch_size)
            if valid_dset:
                self.valid = ODEDataSequence(
                    valid_dset, self.data_ids, batch_size=batch_size if \
                        valid_batch_size is None else valid_batch_size
                    )

        # Testing data
        self.test = test_data

        # Update data ids
        if training:
            self.data_ids = self.train.data_ids
            if self.valid:
                self.data_ids_valid = self.valid.data_ids

        # ODE properties
        self.input_is_list = True if isinstance(self.dim['inp'], (list,tuple)) \
            and len(self.dim['inp']) > 1 else False

        if time_idx not in [0,-1]:
            utils.raise_value_err(
                "`time_idx` passed into `{}` has to be 0 or -1."
                "It's the position (if list) or column (if array) of time "
                "input vector.".format(self.name))
        self.time_idx = time_idx

        self.ode_order = ode_order
        self.ode_func = ode_func
        self.grad_fn = gradients.get_grad_fn(self.ode_order)

    # Handling Data
    ###########################################################################
    def check_data_source(self, data, data_name):
        if data is None:
            utils.raise_value_err("No `{}` passed into `{}`.".format(
                data_name, self.name)
            )
        elif not isinstance(data, dict):
            utils.raise_value_err(
                "`{}` passed into `{}` has to be a `dict`:\n"
                "  - 'ics': initial condition pts (arrays or file path);\n"
                "  - 'pts': additional anchor pts (arrays or file path);\n"
                "  - 'res': collocation pts for residual (arrays or file "
                "path).".format(data_name, self.name)
            )

    def data_as_dict(self, data):
        data = [ self.get_data(data[i], i) for i in self.data_ids ]
        return {self.data_ids[i]: data[i] for i in range(len(self.data_ids))}

    def get_data(self, data, data_id):
        if data is None:
            return None
        if self.from_file:
            # Data from `.csv` file
            if self.col_x is None or self.col_y is None:
                utils.raise_value_err("In `{}` you have to provide both "
                    "`self.col_x` and `self.col_y` to extract data from file."
                    "\nThey should be the doublet (start,end+1) columns "
                    "(or a list of those) of x and y data.".format(self.name))
            df = pd.read_csv(data, sep=',', header=self.header).astype( \
                K.floatx()).values
            if self.preproc_fn is not None:
                df = self.preproc_fn(df)
            x, y = list(
                map(lambda c: self.read_dataframe(df, c), \
                    [self.col_x, self.col_y])
            )
            if any(i.size == 0 for i in x):
                utils.raise_value_err("No input data (see `col_x`) can be "
                    "extracted from file '{}'.".format(data))
            if any(i.size == 0 for i in y):
                if data_id == 'res':
                    y = self.y_fake(x)
                else:
                    utils.raise_value_err("No target data (see `col_y`) can be "
                        "extracted from file '{}'.".format(data))
            return x, y
        else:
            # Data from lists of `numpy.array`
            if isinstance(data, (list,tuple)):
                self.check_data_consistency(data)
                if data_id == 'res':
                    data = data[0], self.y_fake(data[0])
                return data
            else:
                utils.raise_value_err("Data passed into `{}` has to be a "
                    "list or a tuple of arrays: [x, y]".format(self.name))

    def get_num_pts(self, data_type='training', verbose=1):

        def print_fn(dset, data_type, verbose):
            num_pts = {data_id: dset.n_samples[data_id] for data_id in dset.data}
            if verbose:
                utils.print_submain("Number of {} pts:".format(data_type))
                for k, v in num_pts.items():
                    utils.print_submain("  - '%s': %8d" % (k, v))
            return num_pts

        super(ODE, self).get_num_pts(
            data_type=data_type, verbose=verbose, print_fn=print_fn
        )

    def y_fake(self, x):
        return np.zeros((utils.get_cardinality(x),1))

    # Splitting Training/Validation Data
    ###########################################################################
    def get_train_valid(self, data):
        train, valid = {}, {}
        for id_i, data_i in data:
            train[id_i], valid[id_i] = super(ODE, self).get_train_valid(data_i)
        return train, valid

    # Handling Residual Loss
    ###########################################################################
    def res_fn(self, model):
        '''Residual loss function definition.'''

        def residual(inputs, training=True):

            if self.input_is_list:
                ic, t = inputs[self.time_idx-1], inputs[self.time_idx]
            else:
                sizes = [inputs.shape[1]-1, 1]
                if self.time_idx == 0:
                    sizes.reverse()
                ic, t = tf.split(inputs, sizes, axis=1)

            # Evaluate gradients
            grads = self.grad_fn(self, self.ode_order, model, ic, t, training)
            if len(model.outputs_dim) > 1:
                grads = [tf.split(g, model.outputs_dim, axis=1) for g in grads]

            # Evaluate residual
            res_pred = self.ode_func(ic, t, grads)

            return res_pred
        
        return residual
