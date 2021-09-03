import tensorflow as tf
from tensorflow.python.eager        import backprop
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.ops          import math_ops
from tensorflow.python.keras.utils  import losses_utils


def train_step(model, data):
    """Custom train step."""

    x, y = data_adapter.expand_1d(data)

    # Gradient descent step
    losses = {}
    with backprop.GradientTape() as tape:
        for i, data_id in enumerate(model.data_ids):
            losses[data_id] = get_loss(x[i], y[i], model, data_id)
        # Regularization loss
        reg_loss = get_reg_loss(model)
        # Calculate total loss
        tot_loss_weighted = get_tot_loss(model, losses, reg_loss=reg_loss)

    # Update parameters
    model.optimizer.minimize(
        tot_loss_weighted, model.trainable_variables, tape=tape
    )

    # Collect metrics to return
    if model.ode_loss_weights is None:
        total_loss = tot_loss_weighted
        metric = {}
    else:
        # Calculate unweighted total loss
        total_loss = get_tot_loss(model, losses, training=False)
        # Add weighted total loss
        metric = {'tot_loss_weighted': tot_loss_weighted}

    return {**metric, **get_metrics(total_loss, losses, reg_loss=reg_loss)}


def test_step(model, data):
    """Custom test step."""

    x, y = data_adapter.expand_1d(data)

    losses = {}
    for i, data_id in enumerate(model.data_ids_valid):
        losses[data_id] = get_loss(x[i], y[i], model, data_id, training=False)

    # Calculate total loss
    total_loss = get_tot_loss(model, losses, training=False)

    # Collect metrics to return
    return get_metrics(total_loss, losses)


# Util functions
###############################################################################
def get_loss(x, y, model, data_id, training=True):
    if data_id == 'res':
        y_pred = model.residual(x, training=training)
        # if isinstance(y_pred, (list,tuple)):
        #     y = [ tf.zeros_like(y_pred_i) for y_pred_i in y_pred ]
        # else:
        #     y = tf.zeros_like(y_pred)
    else:
        y_pred = model(x, training=training)
    return model.compiled_loss[data_id](y, y_pred)

def get_reg_loss(model):
    if model.losses:
        reg_loss = losses_utils.cast_losses_to_common_dtype(model.losses)
        return math_ops.add_n(reg_loss)

def get_tot_loss(model, losses, reg_loss=None, training=True):
    total_loss = 0.
    if training and reg_loss is not None:
        total_loss = total_loss + reg_loss
    for data_id, loss in losses.items():
        if training and model.ode_loss_weights is not None:
            total_loss = total_loss + model.ode_loss_weights[data_id]*loss
        else:
            total_loss = total_loss + loss
    return total_loss

def get_metrics(total_loss, losses, reg_loss=None):
    metrics = {}
    metrics['tot_loss'] = total_loss
    for data_id, loss in losses.items():
        metrics[data_id + '_loss'] = loss
    if reg_loss is not None:
        metrics['reg_loss'] = reg_loss
    return metrics
