import tensorflow as tf
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.utils  import losses_utils


def multi_obj_train_step(model, data):
    """Custom train step."""

    x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

    # Gradient descent step
    loss, losses = 0., {}
    with backprop.GradientTape() as tape:
        for obj in model.objectives['train']:
            y_pred = model.get_pred(x=x[obj], objective=obj, training=True)
            if model.weighted_loss is not None:
                sample_weight = model.weighted_loss.get_weights(
                    obj,
                    sample_weight=sample_weight
                )
            losses[obj] = model.compiled_loss[obj](
                y[obj],
                y_pred,
                sample_weight=sample_weight
            )
        # Regularization loss
        if model.losses:
            reg_loss = losses_utils.cast_losses_to_common_dtype(model.losses)
            reg_loss = tf.add_n(reg_loss)
            losses.update({'reg': reg_loss})
        # Calculate total loss
        loss = tf.add_n(losses.values())

    # Update parameters
    model.optimizer.minimize(loss, model.trainable_variables, tape=tape)

    # Update loss weights
    if model.weighted_loss is not None:
        model.weighted_loss.update_weights(loss, tape=tape)

    # Collect metrics to return
    metrics = {'loss': loss}
    metrics.update({obj+'_loss': l for obj, l in losses.items()})
    return metrics

def multi_obj_test_step(model, data):
    """Mul test step."""

    x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

    losses = {}
    for obj in model.objectives['valid']:
        y_pred = model.get_pred(x=x[obj], objective=obj)
        sw = sample_weight[obj] if sample_weight else None
        losses[obj] = model.compiled_loss[obj](y[obj], y_pred, sw)

    # Calculate total loss
    loss = tf.add_n(losses.values())

    # Collect metrics to return
    metrics = {'loss': loss}
    metrics.update({obj+'_loss': l for obj, l in losses.items()})
    return metrics
