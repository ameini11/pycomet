from tensorflow.python.keras.engine import data_adapter
from pycomet.data.data import Data
from pycomet.utilities import utils


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
        header=0,
        preproc_fn=None
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

        # Training/Validation data
        if self.training:
            if valid_data:
                self.train = self.get_data(train_data, 'training')
                self.valid = self.get_data(valid_data, 'validation')
            elif validation_split:
                self.train, self.valid = data_adapter.train_validation_split(
                    self.get_data(train_data, 'training'), validation_split
                )
            else:
                self.train = self.get_data(train_data, 'training')

    # Handling Data
    ###########################################################################
    def get_num_pts(self, identifier='training', verbose=1):

        def print_fn(dset, identifier, verbose):
            num_pts = utils.get_cardinality(dset)
            if verbose:
                text = f"Number of {identifier} pts:"
                utils.print_submain(
                    text + ' '*(30-len(text)) + "%10d" % num_pts
                )

        super(DataSet, self).get_num_pts(identifier, verbose, print_fn)
