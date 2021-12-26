import tensorflow as tf

from nltk import flatten
from prode.utilities import utils
from sklearn.model_selection import train_test_split


class Data(object):
    """Data base class."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        training=True,
        batch_size=None,
        from_file=False,
        columns=None,
        header=None,
        preproc_fn=None
    ):
        self.name = utils.get_class_name(self)

        self.dim = dim
        self.training = training

        self.train = None
        self.valid = None
        self.test = None
        self.validation_split = None
        
        self.batch_size = {
            'train': 32,
            'valid': None
        }
        if batch_size != None:
            self.update_batch_size(batch_size)

        # Extra attributes if `data` extracted from file
        self.from_file = from_file
        self.columns = columns
        self.header = header
        self.preproc_fn = preproc_fn

    # Util functions
    ###########################################################################
    def get_num_pts(self, identifier='training', verbose=1, print_fn=None):
        dset = self.train if identifier == 'training' else self.valid
        if dset is not None and print_fn is not None:
            print_fn(dset, identifier, verbose)
        else:
            if verbose:
                utils.print_submain("No {} pts available.".format(identifier))

    # Splitting training/validation data
    ###########################################################################
    def get_train_valid(self, data):
        if data is None:
            return None, None
        else:
            return self.train_valid_split(*data)

    def train_valid_split(self, x, y):
        x, y = list(map(utils.to_list, [x, y]))

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
            x, x_val = x[-1], x_val[-1]
        # Single output model
        if len(y) == 1:
            y, y_val = y[-1], y_val[-1]

        return (x, y), (x_val, y_val)

    # Reading data from file
    ###########################################################################
    def read_datafile(self, file):
        utils.check_file_extension(file, '.cvs')
        if not isinstance(self.columns, dict):
            utils.raise_value_err(
                "`columns` provided in `{}` has to be a `dict` with keys "
                "['x', 'y']. The corresponding values are doublets "
                "(start, end+1) columns (or a list of those if "
                "multi I/O) for extracting input and target data "
                "from file.".format(self.name)
            )
        elif None in self.columns.values():
            utils.raise_value_err(
                "In `columns` dictionary, provided in `{}`, both 'x' and "
                "'y' keys have to be not `None`.".format(self.name)
            )
        df = pd.read_csv(data, sep=',', header=self.header)
        df = df.to_numpy(dtype=K.floatx())
        if self.preproc_fn is not None:
            df = self.preproc_fn(df)
        return [self.read_dataframe(df, self.columns[i]) for i in ['x', 'y']]

    def read_dataframe(self, df, col):
        if any(isinstance(i, (list,tuple)) for i in col):
            return [df[:, col[i][0]:col[i][1]] for i in range(len(col))]
        else:
            return df[:, col[0]:col[1]]

    # Checking data - network consistency
    ###########################################################################
    def check_data_consistency(self, data):
        net_dim = list(self.dim.values())
        data_dim = tf.nest.map_structure(
            lambda z: z.shape[1] if z is not None else None, data
        )
        for i in range(2):
            if data_dim[i] is not None:
                if flatten(data_dim[i]) != flatten(net_dim[i]):
                    utils.raise_value_err(
                        "Data and network I/O dimensions "
                        "not consistent: see `dim`."
                    )

    # Updating default batch sizes
    ###########################################################################
    def update_batch_size(self, batch_size):
        if isinstance(batch_size, dict):
            if all(k in batch_size.keys() for k in ['train', 'valid']):
                self.batch_size = batch_size
            else:
                utils.raise_value_err(
                    "`batch_size` dictionary passed into `{}` has to have "
                    "['train', 'valid'] as keys.".format(self.name)
                )
        elif isinstance(batch_size, int):
            for k in self.batch_size.keys():
                self.batch_size[k] = batch_size
        else:
            utils.raise_value_err(
                "`batch_size` passed into `{}` has to be a "
                "`int` or a `dict`".format(self.name)
            )

