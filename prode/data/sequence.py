import numpy as np
import tensorflow as tf

from tensorflow.python.util.tf_export import keras_export
from prode.utilities import utils


@keras_export('keras.utils.ODEDataSequence')
class ODEDataSequence(tf.keras.utils.Sequence):

    def __init__(self, data, data_ids, batch_size=64, shuffle=True):
        super(ODEDataSequence, self).__init__()

        self.name = utils.get_class_name(self)

        # Check data identities `data_ids`
        self.data_ids = list(data_ids)

        # Check `batch_size` variable
        if not isinstance(batch_size, (dict, int)):
            utils.raise_value_err(
                "`batch_size` passed into `{}` has to be a `int` or `dict`. If `dict`:\n"
                "  - 'ics': initial condition pts batch size;\n"
                "  - 'pts': additional anchor pts batch size;\n"
                "  - 'res': collocation pts batch size (residual).".format(self.name))
        if isinstance(batch_size, int):
            self.batch_size = {data_id: batch_size for data_id in self.data_ids}
        else:
            self.batch_size = batch_size

        # Check `data` variable
        if not isinstance(data, dict):
            utils.raise_value_err(
                "`data` passed into `{}` has to be a `dict`:\n"
                "  - 'ics': initial condition pts dataset;\n"
                "  - 'pts': additional anchor pts dataset;\n"
                "  - 'res': collocation pts dataset (residual).".format(self.name))
        else:
            self.data, self.n_samples = {}, {}
            for k in list(data_ids):
                if k not in data or data[k] is None:
                    # Delete the corresponding `key` entry
                    self.data_ids.remove(k)
                    if k in self.batch_size:
                        self.batch_size.pop(k)
                else:
                    if not isinstance(data[k], (list,tuple)) and len(data[k]) > 2:
                        # Check data type: `[x,y]` or `(x,y)`
                        utils.raise_value_err(
                            "Values passed into `data` dictionary in `{}` must "
                            "be a `list`/`tuple` of `np.ndarray` of length 2: "
                            "(inputs, outputs).".format(self.name))
                    else:
                        self.data[k] = data[k]
                        self.n_samples[k] = utils.get_cardinality(data[k])
                        if k not in self.batch_size:
                            self.batch_size[k] = 32

        # Number of batches for each data id
        self.steps = {k: int(np.ceil(self.n_samples[k]/self.batch_size[k])) for k in self.data_ids}

        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Number of batches in the Sequence.
        Returns:
            The number of batches in the Sequence.
        """
        return max(list(self.steps.values()))


    # Handling batches during each epoch
    ###########################################################################
    @tf.function
    def __getitem__(self, index):
        """Gets batch at position `index`.
        Args:
            index: position of the batch in the Sequence.
        Returns:
            A batch
        """
        x, y = [ xy for xy in zip(*self.sequence[index]) ]
        return x, y


    # Handling data at the end of each epoch
    ###########################################################################
    @tf.function
    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        batches = dict()
        for dset_id, dset in self.data.items():
            if self.shuffle:
                dset = self.shuffle_data(dset, dset_id)
            batches[dset_id] = self.batch_data(dset, dset_id)
        self.sequence = [ batch for batch in zip(*list(batches.values())) ]

    def shuffle_data(self, dset, dset_id):
        indeces = np.random.permutation(self.n_samples[dset_id])
        _dset = []
        for xy in dset:
            if isinstance(xy, (list,tuple)):
                # Multiple I/O
                _dset.append([i[indeces] for i in xy])
            else:
                # Single I/O
                _dset.append(xy[indeces])
        return _dset

    def batch_data(self, dset, dset_id):
        repeat = int(np.ceil(self.__len__()/self.steps[dset_id]))
        _dset = []
        for xy in dset:
            if isinstance(xy, (list,tuple)):
                # Multiple I/O
                spl = [ np.array_split(i, self.steps[dset_id])*repeat for i in xy ]
                _xy = [ spl_i for spl_i in zip(*spl) ]
            else:
                # Single I/O
                _xy = np.array_split(xy, self.steps[dset_id])*repeat
            _dset.append(_xy)
        return [ _dset_i for _dset_i in zip(*_dset) ]
