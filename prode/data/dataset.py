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
        fname_train=None,
        fname_valid=None,
        fname_test=None,
        col_x=None,
        col_y=None,
        standardize=False
    ):
        if x_train is not None:
            self.train_x, self.train_y = x_train, y_train
            self.valid_x, self.valid_y = x_valid, y_valid
            self.test_x,  self.test_y  = x_test,  y_test
        elif fname_train is not None:
            self.train_x, self.train_y = self.load_filedata(fname_train, col_x, col_y)
            self.valid_x, self.valid_y = self.load_filedata(fname_valid, col_x, col_y)
            self.test_x,  self.test_y  = self.load_filedata(fname_test,  col_x, col_y)
        else:
            raise ValueError("No training data.")

        self.scaler_x = None
        if standardize:
            self._standardize()

    def load_filedata(self, filename, col_x, col_y):
        if filename is None:
            return None, None
        else:
            data = np.loadtxt(filename)
            return data[:, col_x], data[:, col_y]

    def transform_inputs(self, x):
        if self.scaler_x is None:
            return x
        return list(map(lambda scaler, xi: scaler.transform(xi), self.scaler_x, x))

    def _standardize(self):
        def standardize_one(x, *args):
            scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)
            x = scaler.fit_transform(x)
            var = [ scaler.transform(arg) if arg is not None else None for arg in args ]
            return scaler, x, var

        self.scaler_x, self.train_x, self.valid_x, self.test_x = \
                    standardize_one(self.train_x, self.valid_x, self.test_x)
