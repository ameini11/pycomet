import tensorflow as tf
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter

from .. import losses as losses_utils


def get_pred_loss(x, y, encoder, model, data_id, training=True):
    ids  = model.data_ids if training else model.data_ids_val
    x_id = tf.boolean_mask(x, encoder == ids[data_id])
    y_id = tf.boolean_mask(y, encoder == ids[data_id])
    if tf.size(x_id) == 0:
        return None, None
    else:
        loss_fn = model.loss_fn[data_id]
        if data_id == 'res':
            return loss_fn(model, x_id, training=training)
        else:
            y_pred = model(x_id, training=training)
            loss   = loss_fn(y_id, y_pred)
            return y_pred, loss


def train_step(model, data):
    """Custom train step."""

    data = data_adapter.expand_1d(data)
    x, y, encoder = data_adapter.unpack_x_y_sample_weight(data)

    # Gradient descent step
    pred, losses = {}, {}
    with backprop.GradientTape() as tape:
        with backprop.GradientTape(persistent=True) as tape_weights:
            for data_id in model.data_ids:
                pred[data_id], losses[data_id] = get_pred_loss(x, y, encoder, model, data_id)
            # Regularization Losses
            reg_loss = losses_utils.get_reg_losses(model)

        # Apply adaptive loss weights technique
        model.adaptive_weights(model, tape=tape_weights, losses=losses, pred=pred)

        # Calculate total loss
        total_loss = reg_loss
        for data_id, loss in losses.items():
            total_loss = total_loss + model.adaptive_weights.weights[data_id] * loss

    # Update parameters
    model.optimizer.minimize(total_loss, model.trainable_variables, tape=tape)

    # Collect metrics to return
    return_metrics = {}
    return_metrics['reg_loss'] = reg_loss
    for data_id, loss in losses.items():
        return_metrics[data_id + '_loss'] = loss

    return return_metrics


def test_step(model, data):
    """Custom test step."""

    data = data_adapter.expand_1d(data)
    x, y, encoder = data_adapter.unpack_x_y_sample_weight(data)
    
    losses = dict()
    for data_id in model.data_ids_val:
        losses[data_id] = get_loss(x, y, encoder, model, data_id, training=False)
    # Regularization Losses
    reg_loss = losses_utils.get_reg_losses(model)

    # Collect metrics to return
    return_metrics = {}
    return_metrics['reg_loss'] = reg_loss
    for data_id, loss in losses.items():
        return_metrics[data_id + '_loss'] = loss

    return return_metrics

