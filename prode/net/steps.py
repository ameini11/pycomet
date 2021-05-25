import tensorflow as tf
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter


@keras_export("keras.utils.unpack_data", v1=[])
def unpack_data(data):
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
        return (data, None, None,)
    elif len(data) == 1:
        return (data[0], None, None)
    elif len(data) == 2:
        return (data[0], data[1], None)
    elif len(data) == 3:
        return (data[0], data[1], data[2])
    else:
        error_msg = ("Data is expected to be in format `x`, `(x,)`, `(x, y)`, "
                     "or `(x, y, sample_weight, ohe)`, found: {}").format(data)
        raise ValueError(error_msg)


@keras_export("keras.utils.masking", v1=[])
def masking(x, ohe):
    if isinstance(x, (list,tuple)):
        x = tuple(map(lambda i: tf.boolean_mask(i, ohe), x))
    else:
        x = tf.boolean_mask(i, ohe)
    return x



def train_step(model, data):
    """Custom train step."""

    data = data_adapter.expand_1d(data)
    x, y, ohe = unpack_data(data)

    # Splitting anchor pts, ICs pts and residual collocation pts
    num_pts, num_ics, num_res = list(map(
        lambda i: tf.math.count_nonzero(ohe[:,i:i+1]), range(3)))
    x_pts, y_pts = list(map(lambda i: masking(i, ohe[:,0:1]), [x,y]))
    x_ics, y_ics = list(map(lambda i: masking(i, ohe[:,1:2]), [x,y]))
    x_res        = masking(x, ohe[:,2:3])

    with backprop.GradientTape() as tape:
        with backprop.GradientTape(persistent=True) as tape_annealing:
            # Loss from anchor pts
            y_pts_pred = self(x_pts, training=True)
            pts_loss = model.compiled_loss(y_pts, y_pts_pred, \
                   regularization_losses=model.losses)
            # Loss from ICs pts
            y_ics_pred = self(x_ics, training=True)
            ics_loss = model.ics_loss_fn(y_ics, y_ics_pred)
            # Loss from residual collocation pts
            res_loss = model.res_loss_fn(model, x_res)
        if model.lr_annealing:
            # Apply learning rate annealing
            model.loss_weights = update_loss_weights(model.loss_weights, \
                tape_annealing, [pts_loss, ics_loss, res_loss])
        # Calculate total loss
        total_loss = model.loss_weights[0]*pts_loss + \
            model.loss_weights[1]*ics_loss + model.loss_weights[2]*res_loss

    # Gradient descent step
    model.optimizer.minimize(total_loss, model.trainable_variables, tape=tape)

    # Update metrics
    model.compiled_metrics.update_state(y_pts, y_pts_pred)

    # Collect metrics to return
    return_metrics = {}
    return_metrics['ics_loss'] = ics_loss
    return_metrics['res_loss'] = res_loss
    for metric in model.metrics:
        result = metric.result()
        if isinstance(result, dict):
            return_metrics.update(result)
        else:
            return_metrics[metric.name] = result
    if 'loss' in return_metrics.keys():
        return_metrics['pts_loss'] = return_metrics.pop('loss')

    return return_metrics


def test_step(model, data):
    """Custom test step."""
    
    data = data_adapter.expand_1d(data)
    x, y, ohe = unpack_data(data)

    y_pred, res_loss = model.res_loss_fn(model, x, training=False)

    # Updates stateful loss metrics.
    model.compiled_loss(y, y_pred, sample_weight, \
                       regularization_losses=model.losses)
    model.compiled_metrics.update_state(y, y_pred, sample_weight)
    
    # Collect metrics to return
    # metr     = {m.name: m.result() for m in model.metrics}
    # res_metr = {'res_loss': res_loss}
    # return {**res_metr, **metr}

    return_metrics = {}
    return_metrics['res_loss'] = res_loss
    for metric in model.metrics:
        result = metric.result()
        if isinstance(result, dict):
            return_metrics.update(result)
        else:
            return_metrics[metric.name] = result
    return return_metrics






# def train_step(model, data):
#     """Custom train step."""

#     data = data_adapter.expand_1d(data)
#     x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

#     with backprop.GradientTape() as tape:
#         y_pred, res_loss = model.res_loss_fn(model, x)
#         loss = model.compiled_loss(y, y_pred, sample_weight, \
#                regularization_losses=model.losses)
#         total_loss = loss + res_loss

#     model.optimizer.minimize(total_loss, \
#                model.trainable_variables, tape=tape)
#     model.compiled_metrics.update_state(y, y_pred, sample_weight)

#     # Collect metrics to return
#     # metr     = {m.name: m.result() for m in model.metrics}
#     # res_metr = {'res_loss': res_loss}
#     # return {**res_metr, **metr}

#     # Collect metrics to return
#     return_metrics = {}
#     return_metrics['res_loss'] = res_loss
#     for metric in model.metrics:
#         result = metric.result()
#         if isinstance(result, dict):
#             return_metrics.update(result)
#         else:
#             return_metrics[metric.name] = result
#     return return_metrics


# def test_step(model, data):
#     """Custom test step."""
    
#     data = data_adapter.expand_1d(data)
#     x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

#     y_pred, res_loss = model.res_loss_fn(model, x, training=False)

#     # Updates stateful loss metrics.
#     model.compiled_loss(y, y_pred, sample_weight, \
#                        regularization_losses=model.losses)
#     model.compiled_metrics.update_state(y, y_pred, sample_weight)
    
#     # Collect metrics to return
#     # metr     = {m.name: m.result() for m in model.metrics}
#     # res_metr = {'res_loss': res_loss}
#     # return {**res_metr, **metr}

#     return_metrics = {}
#     return_metrics['res_loss'] = res_loss
#     for metric in model.metrics:
#         result = metric.result()
#         if isinstance(result, dict):
#             return_metrics.update(result)
#         else:
#             return_metrics[metric.name] = result
#     return return_metrics
