import numpy as np
import tensorflow as tf

from keras.engine import data_adapter
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
        batch_size=None,
        from_file=False,
        columns=None,
        header=None,
        preproc_fn=None
    ):
        super(DataSet, self).__init__(
            dim=dim,
            training=training,
            batch_size=batch_size,
            from_file=from_file,
            columns=columns,
            header=header,
            preproc_fn=preproc_fn
        )

        # Training/Validation data
        if training:
            if valid_data:
                self.train = self.get_data(train_data, 'training')
                self.valid = self.get_data(valid_data, 'validation')
            elif validation_split:
                self.train, self.valid = data_adapter.train_validation_split(
                    self.get_data(train_data, 'training'), validation_split
                )
            else:
                self.train = self.get_data(train_data, 'training')

        # Testing data
        self.test = test_data

    # Handling Data
    ###########################################################################
    def get_data(self, data, identifier):
        if data is None:
            if identifier != 'training':
                return None
            else:
                utils.raise_value_err(
                    "No '{}' data passed into `{}`.".format(
                        identifier, self.name
                    )
                )
        if self.from_file:
            # Data from a `.csv` file
            return self.read_datafile(data)
        else:
            # Data from lists of `numpy.ndarray`
            if isinstance(data, (list,tuple)):
                self.check_data_consistency(data)
                return data
            else:
                utils.raise_value_err(
                    "'{}' data passed into `{}` has to be a list or "
                    "a tuple of arrays: [x, y]".format(
                        identifier, self.name
                    )
                )

    def get_num_pts(self, identifier='training', verbose=1):

        def print_fn(dset, identifier, verbose):
            num_pts = utils.get_cardinality(dset)
            if verbose:
                utils.print_submain(
                    "Number of %s pts:  %10d" % (identifier, num_pts)
                )

        super(DataSet, self).get_num_pts(identifier, verbose, print_fn)
