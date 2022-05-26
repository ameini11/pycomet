import os
import numpy as np

from tensorflow.python.keras.engine import data_adapter
from pycomet.data.sequence import MultiObjData
from pycomet.data.data import Data
from pycomet.data import gradients
from pycomet.utilities import utils


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
        batch_size=None,
        from_file=False,
        columns=None,
        header=None,
        preproc_fn=None
    ):
        super(ODE, self).__init__(
            dim=dim,
            training=training,
            test_data=test_data,
            batch_size=batch_size,
            from_file=from_file,
            columns=columns,
            header=header,
            preproc_fn=preproc_fn
        )

        # Defining multi-objective functions identifiers
        self.obj_list = (
            'ics',  # Initial condition pts
            'pts',  # Anchor pts
            'res'   # Collocation pts (residual)
        )
        self.multi_obj_loss = True
        # Defining multi-objective functions identifiers
        self.objectives = {
            'train': None,
            'valid': None
        }

        # Training/Validation data
        if self.training:
            # Extracting/Manipulating data objects
            self.check_data_source(train_data, 'training')
            if valid_data:
                self.check_data_source(valid_data, 'validation')
                train_dset = self.data_as_dict(train_data)
                valid_dset = self.data_as_dict(valid_data)
            elif validation_split:
                train_dset, valid_dset = data_adapter.train_validation_split(
                    self.data_as_dict(train_data), validation_split
                )
            else:
                train_dset = self.data_as_dict(train_data)
                valid_dset = None

            # Creating multi-objective data sequences 
            self.train = MultiObjData(
                data=train_dset,
                objectives=self.obj_list,
                batch_size=self.batch_size['train']
            )
            if valid_dset is not None:
                self.valid = MultiObjData(
                    data=valid_dset,
                    objectives=self.obj_list,
                    batch_size=self.batch_size['valid'] \
                        or self.batch_size['train']
                )
            # Defining multi-objective functions identifiers
            self.objectives['train'] = self.train.objectives
            self.objectives['valid'] = self.valid.objectives \
                if self.valid is not None else None

        # ODE properties
        if time_idx not in [0,-1]:
            utils.raise_value_err(
                "`time_idx` passed into `{}` has to be 0 or -1. "
                "It's the position (if list) or column (if array) "
                "of time input vector.".format(self.name))
        self.time_idx = time_idx
        self.ode_order = ode_order
        self.ode_func = ode_func
        if self.ode_func is not None:
            self.physics_informed = True

    def build(self, model):
        if self.training:
            self.train.build(model)
            if self.valid is not None:
                self.valid.build(model)
        self.built = True

    # Handling Data
    ###########################################################################
    def check_data_source(self, data, identifier):
        if data is None:
            utils.raise_value_err(
                "No '{}' data passed into `{}`.".format(identifier, self.name)
            )
        elif not isinstance(data, dict):
            utils.raise_value_err(
                "'{}' data passed into `{}` has to be a `dict`:\n"
                "  - 'ics': initial condition pts (arrays or file path);\n"
                "  - 'pts': additional anchor pts (arrays or file path);\n"
                "  - 'res': collocation pts for residual evaluation "
                "(arrays or file path).".format(identifier, self.name)
            )

    def data_as_dict(self, data):
        d = {}
        for obj in self.obj_list:
            if obj in data.keys():
                d[obj] = self.get_data(data[obj], obj)
        return d

    def get_data(self, data, obj):
        if data is None:
            return None
        if self.from_file:
            # Data from a `.csv` file
            data = self.read_datafile(data)
            if any(i.size == 0 for i in data[0]):
                utils.raise_value_err(
                    "No input data (see `columns['x']`) can be extracted "
                    "from file '{}'.".format(os.path.normpath(data))
                )
            if any(i.size == 0 for i in data[1]):
                if obj != 'res':
                    utils.raise_value_err(
                        "No target data (see `columns['y']`) can be extracted "
                        "from file '{}'.".format(os.path.normpath(data))
                    )
            return data
        else:
            # Data from lists of `numpy.array`
            if isinstance(data, (list,tuple)):
                self.check_data_consistency(data)
                if obj == 'res' and len(data) == 1:
                    x = data[0]
                    y = np.zeros((utils.get_cardinality(x),0))
                    data = [x, y]
                return tuple(data)
            else:
                utils.raise_value_err(
                    "Data passed into `{}` has to be a list or "
                    "a tuple of arrays: [x, y]".format(self.name)
                )

    def get_num_pts(self, identifier='training', verbose=1):

        def print_fn(dset, identifier, verbose):
            num_pts = {obj: dset.n_samples[obj] for obj in dset.data}
            if verbose:
                utils.print_submain("Number of {} pts:".format(identifier))
                for k, v in num_pts.items():
                    utils.print_submain("  - '%s': %8d" % (k, v))

        super(ODE, self).get_num_pts(identifier, verbose, print_fn)

    # Handling Residual Loss
    ###########################################################################
    def make_residual_function(self, model):
        '''Residual loss function definition.'''

        def residual(inputs, training=True):

            # Manipulate inputs
            if len(model.inputs) > 1:
                ic, t = inputs[self.time_idx-1], inputs[self.time_idx]
            else:
                t = np.take(inputs, [self.time_idx], axis=1)
                ic = np.delete(inputs, self.time_idx, axis=1)

            # Evaluate gradients
            grads = gradients.eval(
                self.time_idx, self.ode_order, model, ic, t, training
            )

            # Evaluate residual
            res_pred = self.ode_func(ic, t, grads)

            return res_pred
        
        return residual
