import numpy as np
import tensorflow as tf

from varname import nameof
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
        validation_split=None,
        from_file=False,
        col_x=None,
        col_y=None,
        header=None,
        preproc_fn=None,
        batch_size=32,
        valid_batch_size=None
    ):
        super(DataSet, self).__init__(dim=dim, training=training)

        # Only if data from file
        self.from_file = from_file
        self.col_x = col_x
        self.col_y = col_y
        self.header = header
        self.preproc_fn = preproc_fn

        # Training/Validation data
        if training:
            if valid_data:
                self.train = self.get_data(train_data, nameof(train_data))
                self.valid = self.get_data(valid_data, nameof(valid_data))
            elif validation_split:
                self.validation_split = validation_split
                self.train, self.valid = self.get_train_valid(
                    self.get_data(train_data, nameof(train_data))
                )
            else:
                self.train = self.get_data(train_data, nameof(train_data))
                self.valid = None

        # Testing data
        self.test = test_data

        self.batch_size = batch_size
        self.valid_batch_size = valid_batch_size

    # Handling Data
    ###########################################################################
    def get_data(self, data, data_name):
        if data is None:
            if data_name != 'train_data':
                return None
            else:
                utils.raise_value_err(
                    "No training data passed into `{}`.".format(self.name)
                )
        if self.from_file:
            # Data from `.csv` file
            if self.col_x is None or self.col_y is None:
                utils.raise_value_err("In `{}` you have to provide both "
                    "`self.col_x` and `self.col_y` to extract data from file."
                    "\nThey should be the doublet (start,end+1) columns "
                    "(or a list of those) of x and y data.".format(self.name))
            df = pd.read_csv(data, sep=',', header=self.header).astype(
                K.floatx()).values
            if self.preproc_fn is not None:
                df = self.preproc_fn(df)
            return list(
                map(lambda c: self.read_dataframe(df, c), \
                    [self.col_x, self.col_y])
            )
        else:
            # Data from lists of `numpy.array`
            if isinstance(data, (list,tuple)):
                self.check_data_consistency(data)
                return data
            else:
                utils.raise_value_err("Data passed into `{}` has to be a "
                    "list or a tuple of arrays: [x, y]".format(self.name))


    def get_num_pts(self, data_type='training', verbose=1):

        def print_fn(dset, data_type, verbose):
            num_pts = utils.get_cardinality(dset)
            if verbose:
                utils.print_submain(
                    "Number of", data_type, "batches:  %10d" % num_pts
                )
            return num_pts

        super(DataSet, self).get_num_pts(
            data_type=data_type, verbose=verbose, print_fn=print_fn
        )
