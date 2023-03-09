import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.python.keras import backend
from pycomet import operations as ops
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

        self.data_gen = None
        self.physical_mdl = None

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
            utils.raise_value_err(
                "'data_src' has to be a 'dict' with keys "
                "['train', 'valid', 'test']."
            )
        if training and (
            'train' not in self.data_src or self.data_src['train'] is None
        ):
            utils.raise_value_err("No 'train' data passed into 'data_src'.")

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
        if self.from_file:
            if columns is None:
                utils.raise_value_err(
                    f"'columns' passed into '{self.name}' has to be a "
                    "dictionary (structured as 'data_src') with last-level "
                    "keys ['x', 'y', 'sample_weight']."
                )
            elif any([k in columns for k in ('x', 'y', 'sample_weight')]):
                self.columns = tf.nest.map_structure(
                    lambda _: columns, self.data_src
                )
            else:
                self.columns = columns
            # Check array preprocessing functions
            if preproc_fn is None or callable(preproc_fn):
                self.preproc_fn = tf.nest.map_structure(
                    lambda _: preproc_fn, self.data_src
                )
            elif isinstance(preproc_fn, dict):
                self.preproc_fn = preproc_fn
            else:
                utils.raise_value_err(
                    f"'preproc_fn' passed into '{self.name}' has to "
                    "be a dictionary or a callable function."
                )
            # Check consistency between dictionaries
            list_of_dicts = [
                ('data_src', self.data_src),
                ('columns', self.columns),
                ('preproc_fn', self.preproc_fn)
            ]
            utils.check_dicts_consistency(list_of_dicts)
        else:
            self.columns = columns
            self.preproc_fn = preproc_fn

        # Testing data --------------------------------------------------------
        if 'test' in self.data_src:
            test = self.data_src['test']
            if self.from_file and os.path.isdir(test):
                self.test = {}
                for file in os.listdir(test):
                    self.test[Path(file).stem] = self.get_data_from_src(
                        data=os.path.join(test, file), identifier='test'
                    )
            elif isinstance(test, (list,tuple,dict)):
                self.test = tf.nest.map_structure(lambda x: np.array(x), test)
                if not isinstance(self.test, dict):
                    self.test = {
                        'case_' + str(i).zfill(4): dset \
                            for i, dset in enumerate(self.test)
                    }
            else:
                utils.raise_value_err(
                    f"Input 'data_src['test']' passed into `{self.name}` has "
                    "to be a `list`/`tuple`/`dict` of arrays or a `str` "
                    "pointing at the folder containting all the test cases "
                    "saved in `.csv` format."
                )

    # Building Data
    ###########################################################################
    def build(self, model=None):
        self.built = True

    # Handling Data
    ###########################################################################
    def get_data_from_src(self, data, identifier):
        kwargs = {}
        if self.from_file:
            for k in ('columns', 'preproc_fn'):
                attr = getattr(self, k)
                if data is not None and attr is not None:
                    kwargs[k] = attr[identifier]
        return self.extract_data(data=data, **kwargs)

    # Extract data from container
    def extract_data(self, data, columns=None, preproc_fn=None):
        if data is not None:
            if self.from_file:
                return self.read_datafile(data, columns, preproc_fn)
            else:
                # Data from lists of `numpy.ndarray`
                if isinstance(data, (list,tuple)):
                    self.check_data_consistency(data)
                    return tuple(data)
                else:
                    utils.raise_value_err(
                        f"Data passed into `{self.name}` has to be a list "
                        "or a tuple of arrays: ('x', 'y', 'sample_weight')"
                    )

    # Reading data from file
    def read_datafile(self, file, columns, preproc_fn=None):
        utils.check_file_extension(file, '.csv')
        # Read '.csv' file
        df = pd.read_csv(file, sep=',', header=self.header)
        if callable(preproc_fn):
            df = pd.DataFrame(
                data=preproc_fn(df.to_numpy(dtype=backend.floatx())),
                columns=df.columns
            )
        # Collect data from file
        data = []
        for i in ('x', 'y', 'sample_weight'):
            if i in ('x', 'y') and i not in columns.keys():
                utils.raise_value_err(
                    f"'{i}' has not been provided in `columns` dictionary."
                )
            if i in columns.keys():
                data.append(ops.extract_data_from_array(df, columns[i]))
        return tuple(data)

    # # Reading data from file
    # def read_datafile(self, file, columns, preproc_fn=None):
    #     utils.check_file_extension(file, '.csv')
    #     # Read '.csv' file
    #     df = pd.read_csv(file, sep=',', header=self.header)
    #     df = df.to_numpy(dtype=backend.floatx())
    #     if callable(preproc_fn):
    #         df = preproc_fn(df)
    #     # Collect data from file
    #     data = []
    #     for i in ('x', 'y', 'sample_weight'):
    #         if i in ('x', 'y') and i not in columns.keys():
    #             utils.raise_value_err(
    #                 f"'{i}' has not been provided in `columns` dictionary."
    #             )
    #         if i in columns.keys():
    #             data.append(ops.extract_data_from_array(df, columns[i]))
    #     return tuple(data)

    # Checking data - network consistency
    ###########################################################################
    def check_data_consistency(self, data):
        net_dim = list(self.dim.values())
        data_dim = ops.get_struct_shape(data, axis=1)
        for i, io in enumerate(data_dim):
            if i <= 2 and data_dim[i]:
                if tf.nest.flatten(data_dim[i]) != tf.nest.flatten(net_dim[i]):
                    utils.raise_value_err(
                        "Data and network dimensions not "
                        "consistent: check `dim` or data shape."
                    )

    # Updating default batch sizes
    ###########################################################################
    def set_batch_size(self, batch_size):
        if isinstance(batch_size, dict):
            if all([k in batch_size.keys() for k in ('train', 'valid')]):
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

    # Util functions
    ###########################################################################
    def get_num_pts(self, identifier='training', verbose=1, print_fn=None):
        dset = self.train if identifier == 'training' else self.valid
        if dset is not None and print_fn is not None:
            print_fn(dset, identifier, verbose)
        else:
            if verbose:
                utils.print_fn(text=f"No {identifier} pts available.", lev=1)