import numpy as np
import tensorflow as tf

from prode.data.data import Data
from prode.utilities import utils

from sklearn.model_selection import train_test_split
from varname import nameof

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
        source='array',
        train_data=None,
        valid_data=None,
        test_data=None,
        valid_split=None,
        col_x=None,
        col_y=None,
        header=None,
        preproc_fn=None,
        batch_size=32,
        valid_batch_size=None
    ):
        super(DataSet, self).__init__(dim=dim, training=training)

        self.source = source

        # Only if data from file
        self.col_x = col_x
        self.col_y = col_y
        self.header = header
        self.preproc_fn = preproc_fn

        # Training/Validation data
        if training:
            if valid_data:
                self.train = self.get_data(train_data, nameof(train_data))
                self.valid = self.get_data(valid_data, nameof(valid_data))
            elif valid_split:
                self.validation_split = valid_split
                self.train, self.valid = self.get_train_valid(self.get_data(train_data, nameof(train_data)))
            else:
                self.train = self.get_data(train_data, nameof(train_data))
                self.valid = None

        # Testing data
        self.test = test_data
        # if test_data:
        #     self.test = self.get_data(test_data, nameof(test_data))

        # if train_data is not None:
        #     self.train = train_data
        #     self.valid = valid_data
        #     self.test  = test_data
        #     # self.train = self.get_dataset(train_data, batch_size)
        #     # self.valid = self.get_dataset(valid_data, valid_batch_size)
        #     # self.test  = self.get_dataset(test_data)
        # elif file_train is not None:
        #     self.train = self.load_data(file_train)
        #     self.valid = self.load_data(file_valid)
        #     self.test  = self.load_data(file_test)
        #     # self.train = self.load_data(file_train, batch_size=batch_size)
        #     # self.valid = self.load_data(file_valid, batch_size=valid_batch_size)
        #     # self.test  = self.load_data(file_test)
        # elif training:
        #     utils.raise_value_err("No training data passed into `{}`.".format(self.name))

        # self.batch_size       = batch_size
        # self.valid_batch_size = valid_batch_size
        # self.validation_split = valid_split


        self.batch_size       = batch_size
        self.valid_batch_size = valid_batch_size


    # Handling Data
    ###########################################################################
    def get_data(self, data, data_name):
        if data is None:
            if data_name != 'train_data':
                return None
            else:
                utils.raise_value_err("No training data passed into `{}`.".format(self.name))

        if self.source == 'array':
            # Data from lists of `numpy.array`
            if isinstance(data, (list,tuple)):
                return data
            else:
                utils.raise_value_err("`{}` passed into `{}` has to be a list "
                    "or a tuple of arrays: [x, y]".format(data_name, self.name))
        elif self.source  == 'file':
            # Data from `.csv` file
            if self.col_x is None or self.col_y is None:
                utils.raise_value_err("In `{}` you have to provide both "
                    "`self.col_x` and `self.col_y` to extract data from file.\nThey should "
                    "be the doublet 'start'-'end+1' columns (or a list of those) "
                    "of x and y data.".format(self.name))
            data_frame = pd.read_csv(data, sep=',', header=self.header).astype(K.floatx()).values
            if self.preproc_fn is not None:
                data_frame = self.preproc_fn(data_frame)
            if any(isinstance(i, (list,tuple)) for i in self.col_x):
                x_data = []
                for i in range(len(self.col_x)):
                    x_data.append(data_frame[:, self.col_x[i][0]:self.col_x[i][1]])
            else:
                x_data = data_frame[:,self.col_x[0]:self.col_x[1]]
            y_data = data_frame[:,self.col_y[0]:self.col_y[1]]
            return x_data, y_data
        else:
            utils.raise_value_err("`source` passed into `{}` has to "
                "be equal to 'array' or 'file'.".format(data_name, self.name))

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



    # Splitting Training/Validation Data
    ###########################################################################
    def get_train_valid(self, data):
        if data == None:
            return None, None
        else:
            return self.split_data(*data)

    def split_data(self, x, y):
        x = x if isinstance(x, (list,tuple)) else [x]
        y = y if isinstance(y, (list,tuple)) else [y]
        y_len = len(y) if isinstance(y, (list,tuple)) else 1
        data  = (*x, y) if x_len > 1 else (x, y)
        split = train_test_split(*data, test_size=self.validation_split)
        x, x_val = [], []
        for i in range(len(data)):
            if i < x_len:
                x.append(split[i*2])
                x_val.append(split[i*2+1])
            else:
                y, y_val = split[i*2], split[i*2+1]
        if x_len == 1:
            x, x_val = *x, *x_val
        return (x, y), (x_val, y_val)

    @static
    def to_list(x):
        return x if isinstance(x, (list,tuple)) else [x]


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
                utils.print_submain("Number of", data_type, "batches:  %10d" % num_pts)
            return num_pts
        else:
            if verbose:
                utils.print_submain("No", data_type, "pts available.")
            return None