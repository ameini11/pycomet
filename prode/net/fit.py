from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.framework import ops


@keras_export("keras.utils.unpack_x_y_sample_weight_ohe", v1=[])
def unpack_x_y_sample_weight_ohe(data):
  """Unpacks user-provided data tuple.
  This is a convenience utility to be used when overriding
  `Model.train_step`, `Model.test_step`, or `Model.predict_step`.
  This utility makes it easy to support data of the form `(x,)`,
  `(x, y)`, or `(x, y, sample_weight)`.
  Standalone usage:
  >>> features_batch = tf.ones((10, 5))
  >>> labels_batch = tf.zeros((10, 5))
  >>> data = (features_batch, labels_batch)
  >>> # `y` and `sample_weight` will default to `None` if not provided.
  >>> x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
  >>> sample_weight is None
  True
  Example in overridden `Model.train_step`:
  ```python
  class MyModel(tf.keras.Model):
    def train_step(self, data):
      # If `sample_weight` is not provided, all samples will be weighted
      # equally.
      x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
      with tf.GradientTape() as tape:
        y_pred = self(x, training=True)
        loss = self.compiled_loss(
          y, y_pred, sample_weight, regularization_losses=self.losses)
        trainable_variables = self.trainable_variables
        gradients = tape.gradient(loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))
      self.compiled_metrics.update_state(y, y_pred, sample_weight)
      return {m.name: m.result() for m in self.metrics}
  ```
  Args:
    data: A tuple of the form `(x,)`, `(x, y)`, or `(x, y, sample_weight)`.
  Returns:
    The unpacked tuple, with `None`s for `y` and `sample_weight` if they are not
    provided.
  """
  if not isinstance(data, tuple):
    return (data, None, None, None)
  elif len(data) == 1:
    return (data[0], None, None, None)
  elif len(data) == 2:
    return (data[0], data[1], None, None)
  elif len(data) == 3:
    return (data[0], data[1], data[2], None)
  elif len(data) == 4:
    return (data[0], data[1], data[2], data[3])
  else:
    error_msg = ("Data is expected to be in format `x`, `(x,)`, `(x, y)`, "
                 "`(x, y, sample_weight)`, or `(x, y, sample_weight, ohe)`, found: {}").format(data)
    raise ValueError(error_msg)



def _disallow_inside_tf_function(method_name):
  if ops.inside_function():
    error_msg = (
        'Detected a call to `Model.{method_name}` inside a `tf.function`. '
        '`Model.{method_name} is a high-level endpoint that manages its own '
        '`tf.function`. Please move the call to `Model.{method_name}` outside '
        'of all enclosing `tf.function`s. Note that you can call a `Model` '
        'directly on `Tensor`s inside a `tf.function` like: `model(x)`.'
    ).format(method_name=method_name)
    raise RuntimeError(error_msg)


def fit(self,
          ohe=None,
          val_ohe=None,
          x=None,
          y=None,
          batch_size=None,
          epochs=1,
          verbose='auto',
          callbacks=None,
          validation_split=0.,
          validation_data=None,
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          steps_per_epoch=None,
          validation_steps=None,
          validation_batch_size=None,
          validation_freq=1,
          max_queue_size=10,
          workers=1,
          use_multiprocessing=False):

    base_layer.keras_api_gauge.get_cell('fit').set(True)
    # Legacy graph support is contained in `training_v1.Model`.
    version_utils.disallow_legacy_graph('Model', 'fit')
    self._assert_compile_was_called()
    self._check_call_args('fit')
    _disallow_inside_tf_function('fit')

    if verbose == 'auto':
      if self.distribute_strategy._should_use_with_coordinator:  # pylint: disable=protected-access
        verbose = 2  # Default to epoch-level logging for PSStrategy.
      else:
        verbose = 1  # Default to batch-level logging otherwise.

    if validation_split:
      # Create the validation data using the training data. Only supported for
      # `Tensor` and `NumPy` input.
      (x, y, sample_weight, ohe), validation_data = (
          data_adapter.train_validation_split(
              (x, y, sample_weight, ohe), validation_split=validation_split))

    if validation_data:
      val_x, val_y, val_sample_weight, val_ohe = (
          unpack_x_y_sample_weight_ohe(validation_data))

    if self.distribute_strategy._should_use_with_coordinator:  # pylint: disable=protected-access
      self._cluster_coordinator = cluster_coordinator.ClusterCoordinator(
          self.distribute_strategy)

    with self.distribute_strategy.scope(), \
         training_utils.RespectCompiledTrainableState(self):
      # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
      data_handler = data_adapter.get_data_handler(
          x=x,
          y=y,
          sample_weight=sample_weight,
          batch_size=batch_size,
          steps_per_epoch=steps_per_epoch,
          initial_epoch=initial_epoch,
          epochs=epochs,
          shuffle=shuffle,
          class_weight=class_weight,
          max_queue_size=max_queue_size,
          workers=workers,
          use_multiprocessing=use_multiprocessing,
          model=self,
          steps_per_execution=self._steps_per_execution)

      # Container that configures and calls `tf.keras.Callback`s.
      if not isinstance(callbacks, callbacks_module.CallbackList):
        callbacks = callbacks_module.CallbackList(
            callbacks,
            add_history=True,
            add_progbar=verbose != 0,
            model=self,
            verbose=verbose,
            epochs=epochs,
            steps=data_handler.inferred_steps)

      self.stop_training = False
      self.train_function = self.make_train_function()
      self._train_counter.assign(0)
      callbacks.on_train_begin()
      training_logs = None
      # Handle fault-tolerance for multi-worker.
      # TODO(omalleyt): Fix the ordering issues that mean this has to
      # happen after `callbacks.on_train_begin`.
      data_handler._initial_epoch = (  # pylint: disable=protected-access
          self._maybe_load_initial_epoch_from_ckpt(initial_epoch))
      logs = None
      for epoch, iterator in data_handler.enumerate_epochs():
        self.reset_metrics()
        callbacks.on_epoch_begin(epoch)
        with data_handler.catch_stop_iteration():
          for step in data_handler.steps():
            with trace.Trace(
                'train',
                epoch_num=epoch,
                step_num=step,
                batch_size=batch_size,
                _r=1):
              callbacks.on_train_batch_begin(step)
              tmp_logs = self.train_function(iterator)
              if data_handler.should_sync:
                context.async_wait()
              logs = tmp_logs  # No error, now safe to assign to logs.
              end_step = step + data_handler.step_increment
              callbacks.on_train_batch_end(end_step, logs)
              if self.stop_training:
                break

        logs = tf_utils.sync_to_numpy_or_python_type(logs)
        if logs is None:
          raise ValueError('Expect x to be a non-empty array or dataset.')
        epoch_logs = copy.copy(logs)

        # Run validation.
        if validation_data and self._should_eval(epoch, validation_freq):
          # Create data_handler for evaluation and cache it.
          if getattr(self, '_eval_data_handler', None) is None:
            self._eval_data_handler = data_adapter.get_data_handler(
                x=val_x,
                y=val_y,
                sample_weight=val_sample_weight,
                batch_size=validation_batch_size or batch_size,
                steps_per_epoch=validation_steps,
                initial_epoch=0,
                epochs=1,
                max_queue_size=max_queue_size,
                workers=workers,
                use_multiprocessing=use_multiprocessing,
                model=self,
                steps_per_execution=self._steps_per_execution)
          val_logs = self.evaluate(
              x=val_x,
              y=val_y,
              sample_weight=val_sample_weight,
              batch_size=validation_batch_size or batch_size,
              steps=validation_steps,
              callbacks=callbacks,
              max_queue_size=max_queue_size,
              workers=workers,
              use_multiprocessing=use_multiprocessing,
              return_dict=True,
              _use_cached_eval_dataset=True)
          val_logs = {'val_' + name: val for name, val in val_logs.items()}
          epoch_logs.update(val_logs)

        callbacks.on_epoch_end(epoch, epoch_logs)
        training_logs = epoch_logs
        if self.stop_training:
          break

      # If eval data_hanlder exists, delete it after all epochs are done.
      if getattr(self, '_eval_data_handler', None) is not None:
        del self._eval_data_handler
      callbacks.on_train_end(logs=training_logs)
      return self.history