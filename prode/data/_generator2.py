import numpy as np
import tensorflow as tf

from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras          import backend as K
from prode import utils


@keras_export('keras.utils.ODEDataSequence')
class ODEDataSequence(tf.keras.utils.Sequence):

    def __init__(self, data, data_ids, batch_size=32, shuffle=True):
        super(ODEDataSequence, self).__init__()

        self.name = utils.get_class_name(self)

        # Check data identities `data_ids`
        if isinstance(data_ids, dict):
            self.data_ids = data_ids
        else:
            self.data_ids = {data_ids[i]: i+1 for i in range(len(data_ids))}

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
                "`data` passed into `{}` has to be a `int` or `dict`. If `dict`:\n"
                "  - 'ics': initial condition pts dataset;\n"
                "  - 'pts': additional anchor pts dataset;\n"
                "  - 'res': collocation pts dataset (residual).".format(self.name))
        else:
            self.data, self.n_samples = {}, {}
            for k, data_i in data.items():
                if data_i is None:
                    # Delete the corresponding `key` entry
                    assert k in self.data_ids and k in self.batch_size
                    self.data_ids   = self.data_ids.pop(k)
                    self.batch_size = self.batch_size.pop(k)
                else:
                    if not isinstance(data_i, (list,tuple)) and len(data_i) > 2:
                        # Check data type: `tf.data.Dataset([x,y])` or `[x,y]` or `(x,y)`
                        utils.raise_value_err(
                            "Values passed into `data` dictionary in `{}` must "
                            "be a `list`/`tuple` of `np.ndarray` of length 2: "
                            "(inputs, outputs).".format(self.name))
                    else:
                        self.data[k]      = data_i
                        self.n_samples[k] = self.get_num_sample(data_i[0])

        # Number of batches for each data id
        self.steps = {k: int(np.ceil(self.n_samples[k]/self.batch_size[k])) for k in self.data_ids}

        self.shuffle = shuffle
        self.on_epoch_end()

    def get_num_sample(self, x):
        if isinstance(x, (list,tuple)):
            return x[0].shape[0]
        else:
            return x.shape[0]

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
        return self.get_next_batch(index)

    def get_next_batch(self, index):
        for i, (k, idx) in enumerate(self.data_ids.items()):
            _x, _y = self.batches[k][index]
            _encoder = np.ones((self.batch_size[k],1), dtype=np.int64) * idx
            if i == 0:
                x, y, encoder = _x, _y, _encoder
            else:
                x, y, encoder = self._concat([x, y, encoder], [_x, _y, _encoder])
        return x, y, encoder

    def _concat(self, data, data_new):
        for i in range(len(data)):
            if isinstance(data[i], (list,tuple)):
                data_new[i] = list(map(lambda i,j: tf.concat((i,j), 0), data[i], data_new[i]))
            else:
                data_new[i] = tf.concat((data[i], data_new[i]), 0)
        return data_new

    # Handling data at the end of each epoch
    ###########################################################################
    @tf.function
    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        self.batches = dict()
        for k, dset in self.data.items():
            repeat = int(np.ceil(self.__len__()/self.steps[k]))
            self.batches[k] = self.batch_data(self.shuffle_data(dset), k, repeat)

    def shuffle_data(self, dset):
        dset_shuffled = []
        for dset_i in dset:
            if isinstance(dset_i, (list,tuple)):
                _dset_i = []
                for i in dset_i:
                    np.random.shuffle(i)
                    _dset_i.append(i)
                dset_shuffled.append(_dset_i)
            else:
                np.random.shuffle(dset_i)
                dset_shuffled.append(dset_i)
        return dset_shuffled

    def batch_data(self, dset, dset_id, repeat):
        steps = self.steps[dset_id]
        dset_batched = []
        for dset_i in dset:
            if isinstance(dset_i, (list,tuple)):
                dset_split = [np.array_split(i, steps)*repeat for i in dset_i]
                _dset_i = [ dset_split_zip for dset_split_zip in zip(*dset_split) ]
            else:
                _dset_i = np.array_split(dset_i, steps)*repeat
            dset_batched.append(_dset_i)
        return [ dset_batched_zip for dset_batched_zip in zip(*dset_batched) ]

    # def batch_data(self, dset, dset_id, repeat):
    #     indeces = [ self.batch_size[dset_id]*(i+1) for i in range(self.steps[dset_id]-1) ]
    #     dset_batched = []
    #     for dset_i in dset:
    #         if isinstance(dset_i, (list,tuple)):
    #             dset_split = [np.split(i, indeces)*repeat for i in dset_i]
    #             _dset_i = [ dset_split_zip for dset_split_zip in zip(*dset_split) ]
    #         else:
    #             _dset_i = np.split(dset_i, indeces)*repeat
    #         dset_batched.append(_dset_i)
    #     return [ dset_batched_zip for dset_batched_zip in zip(*dset_batched) ]
