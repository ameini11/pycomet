import tensorflow as tf
from tensorflow.python.eager        import backprop
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.ops          import math_ops
from tensorflow.python.keras.utils  import losses_utils


def train_step(model, data):
    """Custom train step."""

    x, y = data_adapter.expand_1d(data)

    # Gradient descent step
    pred, losses = {}, {}
    with backprop.GradientTape() as tape:
        with backprop.GradientTape(persistent=True) as tape_weights:
            for i, data_id in enumerate(model.data_ids):
                pred[data_id], losses[data_id] = get_pred_loss(x[i], y[i], model, data_id)
            # Regularization loss
            reg_loss = get_reg_loss(model)

        # Apply adaptive loss weights technique
        model.adaptive_weights(model, tape=tape_weights, losses=losses, pred=pred)

        # Calculate total loss
        total_loss = get_tot_loss(reg_loss, losses)

    # Update parameters
    model.optimizer.minimize(total_loss, model.trainable_variables, tape=tape)

    # Collect metrics to return
    return get_metrics(total_loss, reg_loss, losses)


def test_step(model, data):
    """Custom test step."""

    x, y = data_adapter.expand_1d(data)
    
    pred, losses = {}, {}
    for i, data_id in enumerate(model.data_ids_val):
        pred[data_id], losses[data_id] = get_pred_loss(x[i], y[i], model, data_id, training=False)
    # Regularization loss
    reg_loss = get_reg_loss(model)

    # Calculate total loss
    total_loss = get_tot_loss(reg_loss, losses)

    # Collect metrics to return
    return get_metrics(total_loss, reg_loss, losses)


# Util functions
###############################################################################
def get_pred_loss(x, y, model, data_id, training=True):
    loss_fn = model.loss_fns[data_id]
    if data_id == 'res':
        return loss_fn(model, x, training=training)
    else:
        y_pred = model(x, training=training)
        loss   = loss_fn(y, y_pred)
        return y_pred, loss

def get_reg_loss(model):
    if model.losses:
        reg_loss = losses_utils.cast_losses_to_common_dtype(model.losses)
        return math_ops.add_n(reg_loss)

def get_tot_loss(reg_loss, losses):
    total_loss = 0.
    if reg_loss:
        total_loss = total_loss + reg_loss
    for data_id, loss in losses.items():
        total_loss = total_loss + loss
    return total_loss

def get_metrics(total_loss, reg_loss, losses):
    metrics = {}
    metrics['tot_loss'] = total_loss
    if reg_loss:
        metrics['reg_loss'] = reg_loss
    for data_id, loss in losses.items():
        metrics[data_id + '_loss'] = loss
    return metrics
