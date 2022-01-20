import tensorflow as tf

from prode.utilities import utils


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
        self.built = False

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

        # Defining multi-objective loss function identifiers
        self.multi_obj_loss = False
        self.objectives = None

    def build(self, model):
        self.built = True

    # Util functions
    ###########################################################################
    def get_num_pts(self, identifier='training', verbose=1, print_fn=None):
        dset = self.train if identifier == 'training' else self.valid
        if dset is not None and print_fn is not None:
            print_fn(dset, identifier, verbose)
        else:
            if verbose:
                utils.print_submain("No {} pts available.".format(identifier))
        print("")

    # Reading data from file
    ###########################################################################
    def read_datafile(self, file):
        utils.check_file_extension(file, '.cvs')
        if not isinstance(self.columns, dict):
            utils.raise_value_err(
                "`columns` provided in `{}` has to be a `dict` with keys "
                "['x', 'y']. The corresponding values are doublets "
                "(start, end) columns (or a list of those if "
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
        data_dim = utils.get_struct_shape(data, axis=1)
        for i, IO in enumerate(self.dim.keys()):
            if data_dim[i]:
                if tf.nest.flatten(data_dim[i]) != tf.nest.flatten(net_dim[i]):
                    utils.raise_value_err(
                        "Data and network '{}' dimensions not "
                        "consistent: check `dim` or data shape.".format(IO)
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
