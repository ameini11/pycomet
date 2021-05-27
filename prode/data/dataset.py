import numpy as np
from sklearn import preprocessing

from .data import Data


class DataSet(Data):
    """Fitting Data set.

    Args:
        col_x: List of integers.
        col_y: List of integers.
    """

    def __init__(
        self,
        x_train=None,
        y_train=None,
        x_valid=None,
        y_valid=None,
        x_test=None,
        y_test=None,
        file_train=None,
        file_valid=None,
        file_test=None,
        col_x=None,
        col_y=None):
        super(DataSet, self).__init__()
        if x_train is not None:
            self.train = x_train, y_train
            self.valid = x_valid, y_valid
            self.test  = x_test,  y_test
        elif fname_train is not None:
            self.train = self.load_data(file_train, col_x, col_y)
            self.valid = self.load_data(file_valid, col_x, col_y)
            self.test  = self.load_data(file_test,  col_x, col_y)
        else:
            raise ValueError("[PrODE]: No training data passed into `{}`.".format(self.__name__))

    def load_data(self, filename, col_x, col_y):
        if filename is None:
            return None, None
        if col_x is None or col_y is None:
            raise ValueError('[PrODE]: In `{}` you have to provide `col_x` and '
                '`col_y` to extract data from file.'.format(self.__name__))
        data = pd.read_csv(filename, sep=',', header=None).values
        y_data = data[:, col_y[0]:col_y[1]]
        if any(isinstance(i, list) for i in col_x):
            x_data = []
            for i in range(len(col_x)):
                x_data.append(data[:, col_x[i][0]:col_x[i][1]])
        else:
            x_data = data[:, col_x[0]:col_x[1]], data[:, col_y[0]:col_y[1]]
        return x_data, y_data
