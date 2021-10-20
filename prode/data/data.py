import tensorflow as tf

from prode.utilities import utils
from sklearn.model_selection import train_test_split


class Data(object):
    """Data base class."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        training=True
    ):
        self.name = utils.get_class_name(self)

        self.dim = dim
        self.training = training

        self.train = None
        self.valid = None
        self.test = None
        self.validation_split = None
        
        self.batch_size = None
        self.valid_batch_size = None
        
        self.data_ids = None
        self.data_ids_valid = None

    # Util functions
    ###########################################################################
    def get_num_pts(self, data_type='training', verbose=1, print_fn=None):
        if data_type == 'training':
            dset = self.train
        elif data_type == 'validation':
            dset = self.valid
        else:
            utils.raise_value_err(
                "`data_type` passed into `{}.get_num_pts` "
                "method must be equal to 'training' or "
                "'validation'.".format(self.name))
        if dset:
            return print_fn(dset, data_type, verbose) if print_fn else None
        else:
            if verbose:
                utils.print_submain("No {} pts available.".format(data_type))
            return None

    # Splitting Training/Validation Data
    ###########################################################################
    def get_train_valid(self, data):
        if data is None:
            return None, None
        else:
            return self.split_data(*data)

    def train_valid_split(self, x, y):
        x, y = list(map(self.to_list, [x, y]))

        # Split arrays into random train and validation subsets
        data = (*x, *y)
        split = train_test_split(*data, test_size=self.validation_split)

        # Define `x` and `y` indeces for extracting train and validation subsets
        x_idx = list(zip(*[(i*2, i*2+1) for i in range(len(x))]))
        y_idx = list(zip(*[(i*2, i*2+1) for i in range(len(x),len(x)+len(y))]))

        # Extract train and validation subsets
        x, x_val = [[split[i] for i in x_idx[j]] for j in [0,1]]
        y, y_val = [[split[i] for i in y_idx[j]] for j in [0,1]]

        # Single input model
        if len(x) == 1:
            x, x_val = x[0], x_val[0]
        # Single output model
        if len(y) == 1:
            y, y_val = y[0], y_val[0]

        return (x, y), (x_val, y_val)

    def to_list(self, x):
        return list(x) if isinstance(x, (list,tuple)) else [x]

    def read_dataframe(self, df, col):
        if any(isinstance(i, (list,tuple)) for i in col):
            return [ df[:, col[i][0]:col[i][1]] for i in range(len(col)) ]
        else:
            return df[:, col[0]:col[1]]

    def check_data_consistency(self, data):
        msg = "Inconsistency between data and " \
            "network I/O dimensions (see `dim`)."
        dim = list(self.dim.values())
        for i, xy in enumerate(data):
            if isinstance(xy, (list,tuple)):
                if isinstance(dim[i], (list,tuple)):
                    if not len(xy) == len(dim[i]):
                        utils.raise_value_err(msg)
                    for j in range(len(xy)):
                        if not xy[j].shape[1] == dim[i][j]:
                            utils.raise_value_err(msg)
                else:
                    utils.raise_value_err(msg)
            else:
                if isinstance(dim[i], int):
                    if not xy.shape[1] == dim[i]:
                        utils.raise_value_err(msg)

    def check_size(self, data):
        msg = "Inconsistency between data and " \
            "network I/O dimensions (see `dim`)."
        dim = list(self.dim.values())
        for i, xy in enumerate(data):
            if isinstance(xy, (list,tuple)):
                if isinstance(dim[i], (list,tuple)):
                    if not len(xy) == len(dim[i]):
                        utils.raise_value_err(msg)
                    for j in range(len(xy)):
                        if not xy[j].shape[1] == dim[i][j]:
                            utils.raise_value_err(msg)
                else:
                    utils.raise_value_err(msg)
            else:
                if isinstance(dim[i], int):
                    if not xy.shape[1] == dim[i]:
                        utils.raise_value_err(msg)
