import tensorflow as tf

from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras import callbacks as cb
from pycomet import operations as ops
from pycomet import utils


@keras_export('keras.callbacks.VariablesTracker')
class VariablesTracker(cb.CSVLogger):

    def __init__(
        self,
        var_names=[],
        log_dir=None,
        filename=''
    ):
        super(VariablesTracker, self).__init__(filename)

        self.var_names = tf.nest.flatten(var_names)

        self.log_dir = log_dir
        if self.log_dir is not None:
            self.log_var = tf.summary.create_file_writer(
                self.log_dir + '/scalars/tracked_variables'
            )
        self.tracked_variables = {}

    def on_train_begin(self, logs=None):
        print(self.model.trainable_variables)
        variables = {
            var.name.split(':')[0]: var \
                for var in self.model.trainable_variables
        }
        for name in self.var_names:
            if name not in variables.keys():
                utils.warning(
                    f"Not found 'tf.Variable' with name {name} in "
                    "'model.trainable_variables': it can't be tracked."
                )
                self.var_names.remove(name)
            else:
                self.tracked_variables[name] = variables[name]
        super(VariablesTracker, self).on_train_begin(logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = {}
        if len(self.tracked_variables) > 0:
            logs = {name: var.numpy() for name, var in self.tracked_variables}
        if self.log_dir is not None:
            with self.log_var.as_default():
                for name, value in logs.items():
                    tf.summary.scalar(name=name, data=value, step=epoch)
        super(VariablesTracker, self).on_epoch_end(epoch, logs)
