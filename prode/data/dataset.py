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
            self.train = train_data
            self.valid = valid_data
            self.test  = test_data
            # self.train = self.get_dataset(train_data, batch_size)
            # self.valid = self.get_dataset(valid_data, valid_batch_size)
            # self.test  = self.get_dataset(test_data)
        elif file_train is not None:
            self.train = self.load_data(file_train)
            self.valid = self.load_data(file_valid)
            self.test  = self.load_data(file_test)
            # self.train = self.load_data(file_train, batch_size=batch_size)
            # self.valid = self.load_data(file_valid, batch_size=valid_batch_size)
            # self.test  = self.load_data(file_test)
        elif training:
            utils.raise_value_err("No training data passed into `{}`.".format(self.name))

        self.batch_size       = batch_size
        self.valid_batch_size = valid_batch_size
        self.validation_split = valid_split

        if valid:
            self.train = self.get_dataset(train_data, batch_size)
            self.valid = self.get_dataset(valid_data, valid_batch_size)
        elif valid_split:
            self.validation_split = valid_split
            self.train, self.valid = get_train_valid(self.data_as_dict(train_data))
        else:
            train_dset = self.data_as_dict(train_data)
            valid_dset = None

        self.batch_size       = batch_size
        self.valid_batch_size = valid_batch_size
        self.validation_split = valid_split


    # Handling Data
    ###########################################################################
    def get_data(self, data, var_name, data_id):
        if data is None or data['data_source'] is None:
            return None
        if data['data_source'] == 'array':
            # Data from lists of `numpy.array`
            if isinstance(data[data_id], (list,tuple)):
                return data[data_id]
            elif data_id == 'res':
                shape = [ data[data_id].shape[0], self.dim['out'] ]
                return data[data_id], np.empty(shape)
            else:
                utils.raise_value_err("`{}[{}]` passed into `{}` has to be a list "
                    "or a tuple of arrays: [x, y]".format(var_name, data_id, self.name))
        elif data['data_source'] == 'file':
            # Data from `.csv` file
            if self.col_x is None or self.col_y is None:
                utils.raise_value_err("In `{}` you have to provide both "
                    "`self.col_x` and `self.col_y` to extract data from file.\nThey should "
                    "be the doublet 'start'-'end+1' columns (or a list of those) "
                    "of x and y data.".format(self.name))
            data_frame = pd.read_csv(data[data_id], sep=',', header=self.header).astype(K.floatx()).values
            if self.preproc_fn is not None:
                data_frame = self.preproc_fn(data_frame)
            if any(isinstance(i, (list,tuple)) for i in self.col_x):
                x_data = []
                for i in range(len(self.col_x)):
                    x_data.append(data_frame[:, self.col_x[i][0]:self.col_x[i][1]])
            else:
                x_data = data_frame[:,self.col_x[0]:self.col_x[1]]
            y_data = data_frame[:,self.col_y[0]:self.col_y[1]]
            if y_data.size == 0:
                y_data = np.empty(data[data_id].shape)
            return x_data, y_data
        else:
            utils.raise_value_err("`{}['data_source']` passed into `{}` "
                "has to be equal to 'array' or 'file'.".format(var_name, self.name))

    # def load_data(self, filename):
    # # def load_data(self, filename, batch_size=None):
    #     if filename is None:
    #         return None
    #     if col_x is None or self.col_y is None:
    #         utils.raise_value_err("In `{}` you have to provide `col_x` and "
    #             "`col_y` to extract data from file.".format(self.__name__))
    #     data = pd.read_csv(filename, sep=',', header=self.header).values
    #     if self.preproc_fn is not None:
    #         data = self.preproc_fn(data)
    #     if any(isinstance(i, list) for i in self.col_x):
    #         x_data = []
    #         for i in range(len(self.col_x)):
    #             x_data.append(data[:, self.col_x[i][0]:self.col_x[i][1]])
    #     else:
    #         x_data = data[:, self.col_x[0]:self.col_x[1]], data[:, self.col_y[0]:self.col_y[1]]
    #     y_data = data[:, self.col_y[0]:self.col_y[1]]
    #     return x_data, y_data
    #     # return self.get_dataset(x_data, y_data, batch_size)


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
            num_pts = self.get_cardinality(dset)
            # num_pts = dset.cardinality()
            if verbose:
                utils.print_submain("Number of", data_type, "pts:     %8d" % num_pts)
            return num_pts
        else:
            if verbose:
                utils.print_submain("No", data_type, "pts available.")
            return None