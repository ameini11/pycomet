import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.python.keras import backend
from pycomet import utils
from pathlib import Path


class MultiObjData(Data):
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
        preproc_fn=None,
        obj_list=[]
    ):
        super(DataSet, self).__init__(
            dim=dim,
            training=training,
            test_data=test_data,
            batch_size=batch_size,
            from_file=from_file,
            columns=columns,
            header=header,
            preproc_fn=preproc_fn
        )
        # Attributes for handling multi-objective loss
        self.multi_obj_loss = True
        self.obj_list = obj_list
        self.objectives = {
            'train': None,
            'valid': None
        }
        self.obj_pred_fn = {}
        self.weighted_loss = None

        # Attributes for physics-informed loss
        self.physics_informed = False

    # Handling Data
    ###########################################################################



    def get_data(self, data):
        d = {}
        for obj in self.obj_list:
            if obj in data.keys():
                d[obj] = self.get_data(data[obj], obj)
                super(DataSet, self).__init__(


        return d


    def get_data(self, data, identifier):
        if data is None:
            if identifier == 'training':
                utils._raise_value_err(
                    f"No '{identifier}' data passed into `{self.name}`."
                )
            else:
                return None
        if self.from_file:
            # Data from a `.csv` file
            return tf.nest.map_structure(
                lambda x: self.read_datafile(data, identifier), data
            )
            return self.read_datafile(data, identifier)
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


file = {
    'train': {
        'ics': './data/train/ics.csv',
        'pts': './data/train/pts.csv',
        'res': './data/train/res.csv'
    },
    'valid': {
        'ics': './data/valid/ics.csv',
        'pts': './data/valid/pts.csv',
        'res': './data/valid/res.csv'
    },
    'test': './data/test/ics.csv'
}


    # Reading data from file
    ###########################################################################
    def read_datafile(self, file, identifier=None):
        utils.check_file_extension(file, '.csv')
        if not isinstance(self.columns, dict):
            utils._raise_value_err(
                "`columns` provided in `{}` has to be a `dict` with keys "
                "['train','valid']['x', 'y', 'sample_weight']. The corresponding values are "
                "lists of column indices (or a list of those if "
                "multi I/O) for extracting input and target data "
                "from file.".format(self.name)
            )
        elif None in self.columns.values():
            utils._raise_value_err(
                "In `columns` dictionary, provided in `{}`, all the provided "
                "values have to be not `None`.".format(self.name)
            )
        df = pd.read_csv(file, sep=',', header=self.header)
        df = df.to_numpy(dtype=backend.floatx())
        if self.preproc_fn is not None:
            df = self.preproc_fn(df)
        # Collect data from file
        data = []
        for i in ('x', 'y', 'sample_weight'):
            if i == 'sample_weight' and identifier == 'testing':
                continue
            if i in ('x', 'y') and i not in self.columns.keys():
                utils._raise_value_err(
                    f"'{i}' values have not been "
                    "provided in `columns` dictionary."
                )
            if i in self.columns.keys():
                data.append(self.read_dataframe(df, self.columns[i]))
        return tuple(data)

    def read_dataframe(self, df, col):
        if any(isinstance(c, (list,tuple)) for c in col):
            return [np.take(df, c, axis=1) for c in col]
        else:
            return np.take(df, col, axis=1)

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
