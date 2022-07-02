import numpy as np
import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.python.util.tf_export import keras_export
from pycomet.utilities import utils


@keras_export('keras.utils.MultiObjData')
class MultiObjData(tf.keras.utils.Sequence):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        data,
        objectives,
        batch_size=32,
        shuffle=True
    ):
        super(MultiObjData, self).__init__()

        self.name = utils.get_class_name(self)

        # Objective functions identifiers: ['ics','pts','res']
        self.objectives = list(objectives)

        # Checking `batch_size`
        if not isinstance(batch_size, (dict, int)):
            utils.raise_value_err(
                "`batch_size` passed into `{}` has to be a"
                " `int` or `dict`. If `dict`:\n"
                "  - 'ics': initial condition pts batch size;\n"
                "  - 'pts': additional anchor pts batch size;\n"
                "  - 'res': collocation pts batch size "
                "(residual).".format(self.name))
        if isinstance(batch_size, int):
            self.batch_size = {
                obj: batch_size for obj in self.objectives
            }
        else:
            self.batch_size = batch_size

        # Checking `data`
        if not isinstance(data, dict):
            utils.raise_value_err(
                "`data` passed into `{}` has to be a `dict`:\n"
                "  - 'ics': initial condition pts dataset;\n"
                "  - 'pts': additional anchor pts dataset;\n"
                "  - 'res': collocation pts dataset "
                "(residual).".format(self.name)
            )
        else:
            self.data, self.n_samples = {}, {}
            for obj in list(objectives):
                if obj not in data or data[obj] is None:
                    # Deleting the corresponding `key` entry
                    self.objectives.remove(obj)
                    if obj in self.batch_size:
                        self.batch_size.pop(obj)
                else:
                    if not isinstance(data[obj], (list,tuple)):
                        # Check data type: [x, y, sample_weight]
                        utils.raise_value_err(
                            "Values passed into `data` dictionary in `{}` "
                            "must be a `list`/`tuple` of `np.ndarray`: "
                            "[x, y, sample_weight].".format(self.name)
                        )
                    else:
                        self.data[obj] = list(data[obj])
                        self.n_samples[obj] = utils.get_cardinality(data[obj])

        # Data dimensions
        self.shapes = tf.nest.map_structure(lambda z: z.shape, self.data)

        # Number of batches for each objective data
        self.steps = tf.nest.map_structure(
            lambda x, y: int(np.ceil(x/y)), self.n_samples, self.batch_size
        )

        self.shuffle = shuffle
        self.keep_indeces = False

    # Building
    ###########################################################################
    def build(self, model):
        self.build_targets(model)
        self.on_epoch_end()

    def build_targets(
        self,
        model
    ):
        for obj in self.objectives:
            # Get true and pred values
            y = self.data[obj][1]
            y_pred = model.multi_obj_pred(
                tf.nest.map_structure(
                    lambda d: tf.random.normal((1, d), dtype=K.floatx()),
                    model.input_dim
                ),
                objective=obj
            )
            y_true, y_pred = list(map(
                lambda x: list(x) if isinstance(x, tuple) else x, [y, y_pred]
            ))
            if isinstance(y_pred, tuple):
                y_pred = list(y_pred)
            # Get true and pred feature dimensions
            y_dim = utils.get_struct_shape(y, axis=1)
            y_pred_dim = utils.get_struct_shape(y_pred, axis=1)
            # Check if true and pred dimensions are the same
            if y_dim != y_pred_dim:
                if obj == 'res':
                    self.data[obj][1] = tf.nest.map_structure(
                        lambda d: np.zeros((self.n_samples[obj], d)),
                        y_pred_dim
                    )
                else:
                    utils.raise_value_err(
                        "'{}' data passed into `{}` are inconsistent with the "
                        "network output dimensions.".format(obj, self.name)
                    )
        # Update shapes
        self.shapes = tf.nest.map_structure(lambda z: z.shape, self.data)

    def init_sample_weight(
        self,
        initializer=None,
        trainable=False,
        rank=2
    ):
        for obj in self.objectives:
            # Get initializers
            init_fn = initializer[obj] if initializer else np.ones
            # Define shape
            if rank == 1:
                fn = lambda x: init_fn(shape=(x.shape[0],1), dtype=K.floatx())
            else:
                fn = lambda x: init_fn(shape=x.shape, dtype=K.floatx())
            # Construct weights
            sample_weight = tf.nest.map_structure(fn, self.data[obj][1])
            # Make them tf.Variable if trainable
            if trainable:
                sample_weight =  tf.nest.map_structure(
                    lambda w: tf.Variable(w), sample_weight
                )
            # Add sample weights to the data set
            if len(self.data[obj]) <= 2:
                self.data[obj].append(sample_weight)
            else:
                self.data[obj][2] = sample_weight
        # Update shapes
        self.shapes = tf.nest.map_structure(lambda z: z.shape, self.data)

    # Sequencing
    ###########################################################################
    def __len__(self):
        """Number of batches in the Sequence.
        Returns:
            The number of batches in the Sequence.
        """
        return max(list(self.steps.values()))

    # Handling batches during each epoch --------------------------------------
    def __getitem__(self, index):
        """Gets batch at position `index`.
        Args:
            index: position of the batch in the Sequence.
        Returns:
            A batch
        """
        return utils.zip_dict_of_lists(self.sequence[index])

    # Handling data during training -------------------------------------------
    def on_epoch_end(self):
        """Method called at the end of every epoch."""
        batches = {}
        for obj, dset in self.data.items():
            if self.shuffle:
                dset = self.shuffle_data(obj, dset)
            batches[obj] = self.batch_data(obj, dset)
        self.sequence = utils.zip_dict_of_lists(batches)

    def shuffle_data(self, obj, dset):
        indeces = np.random.permutation(self.n_samples[obj])
        dset = tf.nest.map_structure(
            lambda z: tf.gather(z, indeces).numpy(), dset
        )
        if self.keep_indeces:
            dset.append(indeces.reshape((-1,1)))
        return dset

    def batch_data(self, obj, dset):
        repeat = int(np.ceil(self.__len__()/self.steps[obj]))
        dset = tf.nest.map_structure(
            lambda z: np.array_split(z, self.steps[obj])*repeat, dset
        )
        for i, z in enumerate(dset):
            if self.keep_indeces and i == len(dset)-1:
                break
            else:
                if isinstance(self.shapes[obj][i], list):
                    dset[i] = list(zip(*z))
        dset = list(zip(*dset))
        return dset
