from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras import callbacks as cb
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

        self.tracked_var = []

    def on_train_begin(self, logs=None):
        names = [var.name for var in self.model.extra_trainable_variables]
        for name in self.var_names:
            if name not in names:
                utils.warning(
                    f"Not found 'tf.Variable' with name {name} in "
                    "'model.extra_trainable_variables': it can't be tracked."
                )
                self.var_names.remove(name)
            else:
                for var in self.model.extra_trainable_variables:
                    if var.name == name:
                        self.tracked_var.append(var)
                        break
        super(VariablesTracker, self).on_train_begin(logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = {}
        if len(self.tracked_var) > 0:
            logs = {var.name: var.value() for var in self.tracked_var}
        if self.log_dir is not None:
            with self.log_var.as_default():
                for k, v in logs:
                    tf.summary.scalar(name=k, data=v, step=epoch)
        super(VariablesTracker, self).on_epoch_end(epoch, logs)
