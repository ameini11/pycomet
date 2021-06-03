import numpy as np
import tensorflow as tf

from tensorflow.python.util.tf_export import keras_export
from prode import utils


@keras_export('keras.utils.ODEDataSequence')
class ODEDataSequence(tf.keras.utils.Sequence):

    def __init__(self, dim, data, data_ids, batch_size=32, shuffle=True):
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
                    if not isinstance(data_i, (tf.data.Dataset,list,tuple)):
                        # Check data type: `tf.data.Dataset([x,y])` or `[x,y]` or `(x,y)`
                        utils.raise_value_err(
                            "Values passed into `data` dictionary in `{}` must "
                            "be a `tf.data.Dataset` or a `list`/`tuple`.".format(self.name))
                    elif not isinstance(data_i, tf.data.Dataset):
                        # Transform `list`/`tuple` into `tf.data.Dataset`
                        if len(data_i) > 2:
                            utils.raise_value_err(
                                "`list`/`tuple` passed into `data` dictionary in `{}` must "
                                "be of length 2: (inputs, outputs).".format(self.name))
                        self.data[k], self.n_samples[k] = self.get_data(*data_i)
                    else:
                        self.data[k]      = data_i
                        self.n_samples[k] = data_i.cardinality()

        try:
            self.num_input = len(list(self.data.values())[0].element_spec[0])
        except:
            self.num_input = 1

        # Number of batches for each data id
        self.batches = {k: tf.cast(tf.math.ceil(self.n_samples[k]/self.batch_size[k]), \
            tf.int64) for k in self.data_ids}

        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """Number of batches in the Sequence.
        Returns:
            The number of batches in the Sequence.
        """
        return max(list(self.batches.values()))

    def __getitem__(self, index):
        """Gets batch at position `index`.
        Args:
            index: position of the batch in the Sequence.
        Returns:
            A batch
        """
        return self.sequence[index]

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        iterator = dict()
        for k, dset in self.data.items():
            self.data[k] = dset.shuffle(self.n_samples[k])
            repeat_k = tf.cast(tf.math.ceil(self.__len__()/self.batches[k]), tf.int64)
            iterator[k] = iter(self.data[k].batch( \
                self.batch_size[k]).repeat(repeat_k).take(self.__len__()))
        self.sequence = self.iterator_to_list(iterator)

    def iterator_to_list(self, iterator):
        lst = []
        _has_value = True
        while _has_value:
            for i, (k, idx) in enumerate(self.data_ids.items()):
                batch = iterator[k].get_next_as_optional()
                _has_value = batch.has_value()
                if _has_value:
                    _x, _y   = batch.get_value()
                    _encoder = tf.ones([self.batch_size[k],1], dtype=tf.int64) * idx
                    if i == 0:
                        x, y, encoder = _x, _y, _encoder
                    else:
                        x, y, encoder = self._concat([x, y, encoder], [_x, _y, _encoder])
            lst.append((x, y, encoder))
        return lst

    def _concat(self, data, data_new):
        for i in range(len(data)):
            if isinstance(data[i], (list,tuple)):
                data_new[i] = list(map(lambda i,j: tf.concat((i,j), 0), data[i], data_new[i]))
            else:
                data_new[i] = tf.concat((data[i], data_new[i]), 0)
        return data_new

    def get_data(self, x, y):
        dset_x = self.get_dataset(x)
        dset_y = self.get_dataset(y)
        return tf.data.Dataset.zip((dset_x, dset_y)), dset_x.cardinality()

    def get_dataset(self, x):
        if isinstance(x, list):
            x = tuple(x)
        return tf.data.Dataset.from_tensor_slices(x)