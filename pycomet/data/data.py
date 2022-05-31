import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.python.keras import backend as K
from pycomet.utilities import utils
from pathlib import Path


class Data(object):
    """Data base class."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        training=True,
        test_data=None,
        batch_size=None,
        from_file=False,
        columns=None,
        header=0,
        preproc_fn=None
    ):
        self.name = utils.get_class_name(self)

        self.dim = dim
        self.training = training
        self.built = False

        self.train = None
        self.valid = None
        self.test = None
        self.validation_split = None
        
        self.batch_size = {
            'train': 32,
            'valid': None
        }
        if batch_size != None:
            self.set_batch_size(batch_size)

        # Extra attributes if `data` extracted from file
        self.from_file = from_file
        self.columns = columns
        self.header = header
        self.preproc_fn = preproc_fn

        # Defining multi-objective loss function identifiers
        self.multi_obj_loss = False
        self.objectives = None
        self.physics_informed = False

        # Testing data
        if test_data is None:
            pass
        elif isinstance(test_data, str) and os.path.isdir(test_data):
            self.test = {}
            for file in os.listdir(test_data):
                self.test[Path(file).stem] = self.get_data(
                    os.path.join(test_data, file), 'testing'
                )
        elif isinstance(test_data, (list,tuple,dict)):
            self.test = tf.nest.map_structure(lambda x: np.array(x), test_data)
            if not isinstance(self.test, dict):
                self.test = {
                    f'case_'.zfill(3): dset for i, dset in enumerate(self.test)
                }
        else:
            utils.raise_value_err(
                f"Input 'test_data' passed into `{self.name}` has to be a "
                "`list`/`tuple`/`dict` of arrays or a `str` pointing at the " 
                "folder containting all the test cases saved in `.csv` format."
            )

    # Building Data
    ###########################################################################
    def build(self, model):
        self.built = True

    # Handling Data
    ###########################################################################
    def get_data(self, data, identifier):
        if data is None:
            if identifier == 'training':
                utils.raise_value_err(
                    f"No '{identifier}' data passed into `{self.name}`."
                )
            else:
                return None
        if self.from_file:
            # Data from a `.csv` file
            return self.read_datafile(data, identifier)
        else:
            # Data from lists of `numpy.ndarray`
            if isinstance(data, (list,tuple)):
                self.check_data_consistency(data)
                return tuple(data)
            else:
                utils.raise_value_err(
                    f"'{identifier}' data passed into `{self.name}` has "
                    "to be a list or a tuple of arrays: [x, y]"
                )

    # Util functions
    ###########################################################################
    def get_num_pts(self, identifier='training', verbose=1, print_fn=None):
        dset = self.train if identifier == 'training' else self.valid
        if dset is not None and print_fn is not None:
            print_fn(dset, identifier, verbose)
        else:
            if verbose:
                utils.print_submain("No {} pts available.".format(identifier))

    # Reading data from file
    ###########################################################################
    def read_datafile(self, file, identifier=None):
        utils.check_file_extension(file, '.csv')
        if not isinstance(self.columns, dict):
            utils.raise_value_err(
                "`columns` provided in `{}` has to be a `dict` with keys "
                "['x', 'y', 'sample_weight']. The corresponding values are "
                "doublets (start, end) columns (or a list of those if "
                "multi I/O) for extracting input and target data "
                "from file.".format(self.name)
            )
        elif None in self.columns.values():
            utils.raise_value_err(
                "In `columns` dictionary, provided in `{}`, all the provided "
                "values have to be not `None`.".format(self.name)
            )
        df = pd.read_csv(file, sep=',', header=self.header)
        df = df.to_numpy(dtype=K.floatx())
        if self.preproc_fn is not None:
            df = self.preproc_fn(df)
        # Collect data from file
        data = []
        for i in ('x', 'y', 'sample_weight'):
            if i == 'sample_weight' and identifier == 'testing':
                continue
            if i in ('x', 'y') and i not in self.columns.keys():
                utils.raise_value_err(
                    f"'{i}' values have not been "
                    "provided in `columns` dictionary."
                )
            if i in self.columns.keys():
                data.append(self.read_dataframe(df, self.columns[i]))
        return tuple(data)

    def read_dataframe(self, df, col):
        if any(isinstance(i, (list,tuple)) for i in col):
            return [df[:,col[i][0]:col[i][1]] for i in range(len(col))]
        else:
            return df[:,col[0]:col[1]]

    # Checking data - network consistency
    ###########################################################################
    def check_data_consistency(self, data):
        net_dim = list(self.dim.values())
        data_dim = utils.get_struct_shape(data, axis=1)
        for i, IO in enumerate(data_dim):
            if i <= 2 and data_dim[i]:
                if tf.nest.flatten(data_dim[i]) != tf.nest.flatten(net_dim[i]):
                    utils.raise_value_err(
                        "Data and network '{}' dimensions not "
                        "consistent: check `dim` or data shape.".format(IO)
                    )

    # Updating default batch sizes
    ###########################################################################
    def set_batch_size(self, batch_size):
        if isinstance(batch_size, dict):
            if all(k in batch_size.keys() for k in ['train', 'valid']):
                self.batch_size = batch_size
            else:
                utils.raise_value_err(
                    "`batch_size` dictionary passed into `{}` has to have "
                    "['train', 'valid'] as keys.".format(self.name)
                )
        elif isinstance(batch_size, int):
            for k in self.batch_size.keys():
                self.batch_size[k] = batch_size
        else:
            utils.raise_value_err(
                "`batch_size` passed into `{}` has to be a "
                "`int` or a `dict`".format(self.name)
            )
