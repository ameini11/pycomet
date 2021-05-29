import numpy as np
import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='PrODE', name='ODEDataSequence')
class ODEDataSequence(tf.keras.utils.Sequence):
    def __init__(self, data, data_ids, batch_size=32, shuffle=True):
        super(ODEDataSequence, self).__init__()
        '''Initialization'''
        # Check data identities `data_ids`
        if not isinstance(data_ids, dict):
        	self.data_ids = {data_ids[i]: i+1 for i in range(len(data_ids))}
        else:
        	self.data_ids = data_ids

        # Check `batch_size` variable
        if not isinstance(batch_size, (dict, int)):
            raise ValueError(
                "\n[PrODE]: `batch_size` passed into `{}` has to be a `int` or `dict`.\n"
                "   If `dict`:\n"
                "   >> 'ics': initial condition batch size;\n"
                "   >> 'pts': additional anchor pts batch size;\n"
                "   >> 'res': residual collocation pts batch size.".format(self.__name__))
        if isinstance(batch_size, int):
        	self.batch_size = {data_fam: batch_size for data_fam in self.data_ids}
        else:
        	self.batch_size = batch_size

        # Check `data` variable
        self.data = dict()
        if not isinstance(data, dict):
            raise ValueError(
                "\n[PrODE]: `data` passed into `{}` has to be a `dict`:\n"
                "   >> 'ics': initial condition pts dataset;\n"
                "   >> 'pts': additional anchor pts dataset;\n"
                "   >> 'res': residual collocation pts dataset.".format(self.__name__))
        else:
        	for k, data_i in data:
				if data_i is None:
					# Delete the corresponding `key` entry
					assert k in self.data_ids and k in self.batch_size
					self.data_ids   = self.data_ids.pop(k)
					self.batch_size = self.batch_size.pop(k)
				else:
			        if not isinstance(data_i, (tf.data.Dataset,list,tuple)):
						# Check data type: `tf.data.Dataset([x,y])` or `[x,y]` or `(x,y)`
			            raise ValueError(
			                "\n[PrODE]: Values passed into `data` dictionary in `{}` "
			                "have to be a `tf.data.Dataset` or a `list` or a `tuple`.".format(self.__name__))
			        elif not isinstance(data_i, tf.data.Dataset):
			        	# Transform `list`/`tuple` into `tf.data.Dataset`
			            dset_x = tf.data.Dataset.from_tensor_slices(tuple(data_i[0]))
			            dset_y = tf.data.Dataset.from_tensor_slices(tuple(data_i[1]))
			            self.data[k] = tf.data.Dataset.zip(tuple(dset_x, dset_y))
			        else:
			            self.data[k] = data_i
    	try:
    		self.num_input = len(list(self.data.values())[0].element_spec[0])
    	except:
    		self.num_input = 1

        # Number of samples and batches for each data id
		self.n_samples = {k: self.data[k].cardinality() for k in self.data_ids}
		self.batches   = {k: tf.math.ceil(self.n_samples[k]/self.batch_size[k]) \
								for k in self.data_ids}

        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
		"""Number of batches in the Sequence.
		Returns:
		    The number of batches in the Sequence.
		"""
        return max(list(self.batches.values()))

    def __getitem__(self, *args, **kwargs):
		"""Gets batch at position `index`.
		Args:
		    index: position of the batch in the Sequence.
		Returns:
		    A batch
		"""
		for i, (k, idx) in enumerate(self.data_ids.items()):
			batch = next(self.iterator[k])
            _x, _y   = batch[0], batch[1]
            _encoder = tf.ones([self.batch_size[k], 1]) * idx
            if i == 0:
                x, y, encoder = _x, _y, _encoder
            else:
                x, y, encoder = self._concat((x, y, encoder), (_x, _y, _encoder))
        return x, y, encoder

    def on_epoch_end(self):
		"""Method called at the end of every epoch.
		"""
		self.iterator = dict()
		for k, dset in self.data:
			self.data[k] = dset.shuffle(self.n_samples[k])
			repeat_k = tf.math.ceil(self.__len__()/self.batches[k])
			self.iterator[k] = iter(self.data[k].batch( \
				self.batch_size[k]).repeat(repeat_k).take(self.__len__()))

    def _concat(self, data, data_new):
    	for i in range(len(data)):
    		if isinstance(data[i], (list,tuple)):
    			data_new[i] = list(map(lambda i,j: tf.concat((i,j), 0), data[i], data_new[i]))
    		else:
    			data_new[i] = tf.concat((data[i], data_new[i]), 0)
    	return data_new
