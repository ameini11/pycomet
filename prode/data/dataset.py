import numpy as np
import tensorflow as tf

from prode.data.data import Data
from prode.utilities import utils


class DataSet(Data):
    """Fitting Data set.

    Args:
        col_x: List of integers.
        col_y: List of integers.
    """

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        training=True,
        train_data=None,
        valid_data=None,
        test_data=None,
        file_train=None,
        file_valid=None,
        file_test=None,
        valid_split=None,
        col_x=None,
        col_y=None,
        header=None,
        preproc_fn=None,
        batch_size=32,
        valid_batch_size=32
    ):
        super(DataSet, self).__init__(dim=dim, training=training)

        # Only if data from file
        self.col_x = col_x
        self.col_y = col_y
        self.header = header
        self.preproc_fn = preproc_fn

        if train_data is not None:
            self.train = self.get_dataset(train_data, batch_size)
            self.valid = self.get_dataset(valid_data, valid_batch_size)
            self.test  = self.get_dataset(test_data)
        elif file_train is not None:
            self.train = self.load_data(file_train, batch_size=batch_size)
            self.valid = self.load_data(file_valid, batch_size=valid_batch_size)
            self.test  = self.load_data(file_test)
        elif training:
            utils.raise_value_err("No training data passed into `{}`.".format(self.name))

        self.batch_size       = batch_size
        self.valid_batch_size = valid_batch_size
        self.validation_split = valid_split


    # Handling Data
    ###########################################################################
    def load_data(self, filename, batch_size=None):
        if filename is None:
            return None
        if col_x is None or self.col_y is None:
            utils.raise_value_err("In `{}` you have to provide `col_x` and "
                "`col_y` to extract data from file.".format(self.__name__))
        data = pd.read_csv(filename, sep=',', header=self.header).values
        if self.preproc_fn is not None:
            data = self.preproc_fn(data)
        if any(isinstance(i, list) for i in self.col_x):
            x_data = []
            for i in range(len(self.col_x)):
                x_data.append(data[:, self.col_x[i][0]:self.col_x[i][1]])
        else:
            x_data = data[:, self.col_x[0]:self.col_x[1]], data[:, self.col_y[0]:self.col_y[1]]
        y_data = data[:, self.col_y[0]:self.col_y[1]]
        return self.get_dataset(x_data, y_data, batch_size)


    # Util functions
    ###########################################################################
    def get_num_pts(self, data_type='training', verbose=1):
        if data_type == 'training':
            dset = self.train
        elif data_type == 'validation':
            dset = self.valid
        else:
            utils.raise_value_err("`data_type` passed into `{}.get_num_pts` method "
                "must be equal to 'training' or 'validation'.".format(self.__name__))
        if dset:
            num_pts = dset.cardinality()
            if verbose:
                utils.print_submain("Number of", data_type, "pts:     %8d" % num_pts)
            return num_pts
        else:
            if verbose:
                utils.print_submain("No", data_type, "pts available.")
            return None