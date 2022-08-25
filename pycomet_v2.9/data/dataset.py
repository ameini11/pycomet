from tensorflow.python.keras.engine import data_adapter
from pycomet.data.data import Data
from pycomet import utils


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
        data_src={},
        validation_split=None,
        training=True,
        batch_size=None,
        from_file=False,
        header=0,
        columns=None,
        preproc_fn=None
    ):
        super(DataSet, self).__init__(
            dim=dim,
            data_src=data_src,
            validation_split=validation_split,
            training=training,
            batch_size=batch_size,
            from_file=from_file,
            header=header,
            columns=columns,
            preproc_fn=preproc_fn
        )

        # Training/Validation data
        if self.training:
            if 'valid' in self.data_src and self.data_src['valid'] is not None:
                for k in ('train', 'valid'):
                    setattr(self, k, self.get_data(self.data_src[k], k))
            elif self.validation_split:
                self.train, self.valid = data_adapter.train_validation_split(
                    self.get_data(self.data_src['train'], 'train'),
                    self.validation_split
                )
            else:
                self.train = self.get_data(self.data_src['train'], 'train')

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
