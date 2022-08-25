import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.python.keras import backend
from pycomet import utils
from pathlib import Path


class Data(object):
    """Data base class."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        data_src={},
        validation_split=None,
        training=True,
        batch_size=None,
        from_file=False,
        header=0,
        columns=None,
        preproc_fn=None
    ):
        # Globals -------------------------------------------------------------
        self.name = utils.get_class_name(self)
        self.dim = dim
        self.training = training
        self.built = False
        self.multi_obj_loss = False

        # Data ----------------------------------------------------------------
        self.data_src = data_src
        self.train = None
        self.valid = None
        self.test = None
        self.validation_split = validation_split
        # Check data sources
        if not isinstance(self.data_src, dict) or any([
            k not in ('train', 'valid', 'test') for k in self.data_src.keys()
        ]):
            utils._raise_value_err(
                "'data_src' has to be a 'dict' with keys "
                "['train', 'valid', 'test']."
            )
        if 'train' not in self.data_src or self.data_src['train'] is None:
            utils._raise_value_err("No 'train' data passed into 'data_src'.")

        # Batch sizes ---------------------------------------------------------
        self.batch_size = {
            'train': 32,
            'valid': None
        }
        if batch_size is not None:
            self.set_batch_size(batch_size)

        # Extra attributes if data extracted from file ------------------------
        self.from_file = from_file
        self.header = header
        self.columns = columns
        self.preproc_fn = preproc_fn
        if self.from_file:
            # Check consistency between dictionaries
            list_of_dicts = [
                ('data_src', self.data_src), ('columns', self.columns)
            ]
            # Check array preprocessing functions
            if self.preproc_fn is not None:
                if isinstance(self.preproc_fn, callable):
                    num_elem = len(tf.nest.flatten(self.data_src))
                    self.preproc_fn = tf.nest.pack_sequence_as(
                        self.data_src, [self.preproc_fn] * num_elem
                    )
                elif isinstance(self.preproc_fn, dict):
                    list_of_dicts += [('preproc_fn', self.preproc_fn)]
                else:
                    utils._raise_value_err(
                        f"'preproc_fn' passed into '{self.name}' has to "
                        "be a dictonary or a callable function."
                    )
            utils.check_dicts_consistency(list_of_dicts)

        # Testing data --------------------------------------------------------
        if 'test' in self.data_src:
            test = self.data_src['test']
            if isinstance(test, str) and os.path.isdir(test):
                self.test = {}
                for file in os.listdir(test):
                    self.test[Path(file).stem] = self.get_data(
                        data=os.path.join(test, file),
                        identifier='test',
                        columns=self.columns['test'],
                        preproc_fn=self.preproc_fn['test']
                    )
            elif isinstance(test, (list,tuple,dict)):
                self.test = tf.nest.map_structure(lambda x: np.array(x), test)
                if not isinstance(self.test, dict):
                    self.test = {
                        'case_' + str(i).zfill(4): dset \
                            for i, dset in enumerate(self.test)
                    }
            else:
                utils._raise_value_err(
                    f"Input 'data_src['test']' passed into `{self.name}` has "
                    "to be a `list`/`tuple`/`dict` of arrays or a `str` "
                    "pointing at the folder containting all the test cases "
                    "saved in `.csv` format."
                )

    # Building Data
    ###########################################################################
    def build(self, model):
        self.built = True

    # Handling Data
    ###########################################################################
    def get_data(self, data, identifier, columns=None, preproc_fn=None):
        if data is not None:
            if self.from_file:
                return self.read_datafile(data, columns, preproc_fn)
            else:
                # Data from lists of `numpy.ndarray`
                if isinstance(data, (list,tuple)):
                    self.check_data_consistency(data)
                    return tuple(data)
                else:
                    utils._raise_value_err(
                        f"'{identifier}' data passed into `{self.name}` has "
                        "to be a list or a tuple of arrays: [x, y]"
                    )

    # Reading data from file --------------------------------------------------
    def read_datafile(self, file, columns, preproc_fn=None):
        utils.check_file_extension(file, '.csv')
        # Read '.csv' file
        df = pd.read_csv(file, sep=',', header=self.header)
        df = df.to_numpy(dtype=backend.floatx())
        if preproc_fn is not None and callable(preproc_fn):
            df = preproc_fn(df)
        # Collect data from file
        data = []
        for i in ('x', 'y', 'sample_weight'):
            if i in ('x', 'y') and i not in columns.keys():
                utils._raise_value_err(
                    f"'{i}' has not been provided in `columns` dictionary."
                )
            if i in columns.keys():
                data.append(utils.extract_columns_from_array(df, columns[i]))
        return tuple(data)

    # Checking data - network consistency
    ###########################################################################
    def check_data_consistency(self, data):
        net_dim = list(self.dim.values())
        data_dim = utils.array_ops.get_struct_shape(data, axis=1)
        for i, io in enumerate(data_dim):
            if i <= 2 and data_dim[i]:
                if tf.nest.flatten(data_dim[i]) != tf.nest.flatten(net_dim[i]):
                    utils._raise_value_err(
                        "Data and network '{}' dimensions not "
                        "consistent: check `dim` or data shape.".format(io)
                    )

    # Updating default batch sizes
    ###########################################################################
    def set_batch_size(self, batch_size):
        if isinstance(batch_size, dict):
            if all([k in batch_size.keys() for k in ('train', 'valid')]):
                self.batch_size = batch_size
            else:
                utils._raise_value_err(
                    "`batch_size` dictionary passed into `{}` has to have "
                    "['train', 'valid'] as keys.".format(self.name)
                )
        elif isinstance(batch_size, int):
            for k in self.batch_size.keys():
                self.batch_size[k] = batch_size
        else:
            utils._raise_value_err(
                "`batch_size` passed into `{}` has to be a "
                "`int` or a `dict`".format(self.name)
            )

    # Util functions
    ###########################################################################
    def get_num_pts(self, identifier='training', verbose=1, print_fn=None):
        dset = self.train if identifier == 'training' else self.valid
        if dset is not None and print_fn is not None:
            print_fn(dset, identifier, verbose)
        else:
            if verbose:
                utils._print(text=f"No {identifier} pts available.", lev=1)