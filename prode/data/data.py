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
        
        self.data_ids = None
        
        self.batch_size = None
        self.valid_batch_size = None
        self.validation_split = None


    # Util functions
    ###########################################################################
    def get_cardinality(self, x):
        if isinstance(x, (list,tuple)):
            return self.get_cardinality(x[0])
        else:
            return x.shape[0]

    def get_num_pts(self, fun, data_type='training', verbose=1):
        if data_type == 'training':
            dset = self.train
        elif data_type == 'validation':
            dset = self.valid
        else:
            utils.raise_value_err("`data_type` passed into `{}.get_num_pts` method "
                "must be equal to 'training' or 'validation'.".format(self.__name__))
        if dset:
            return fun(dset, data_type, verbose)
        else:
            if verbose:
                utils.print_submain("No", data_type, "pts available.")
            return None

    # Splitting Training/Validation Data
    ###########################################################################
    def get_train_valid(self, data):
        if data == None:
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
            x, x_val = *x, *x_val
        # Single output model
        if len(y) == 1:
            y, y_val = *y, *y_val

        return (x, y), (x_val, y_val)

    def to_list(self, x):
        return x if isinstance(x, (list,tuple)) else [x]