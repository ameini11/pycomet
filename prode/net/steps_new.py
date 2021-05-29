import tensorflow as tf
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter

from .. import losses as losses_utils




def train_step(model, data):
    """Custom train step."""

    data = data_adapter.expand_1d(data)
    x, y, enc = data_adapter.unpack_x_y_sample_weight(data)

    x_ics, y_ics = get_next_batch(model.data_ics)
    x_pts, y_pts = get_next_batch(model.data_pts)
    x_res, y_res = data_res

    # Gradient descent step
    with backprop.GradientTape() as tape:
        with backprop.GradientTape(persistent=True) as tape_annealing:
            # Loss from ICs pts
            y_ics_pred = model(x_ics, training=True)
            ics_loss   = model.ics_loss_fn(y_ics, y_ics_pred)
            # Loss from anchor pts
            y_pts_pred = model(x_pts, training=True)
            pts_loss   = model.compiled_loss(y_pts, y_pts_pred)
            # Loss from residual collocation pts
            y_res_pred, res_loss = model.res_loss_fn(model, x_res, training=True)
            # Regularization Losses
            reg_loss   = losses_utils.get_reg_losses(model)

        losses = tf.convert_to_tensor([res_loss, ics_loss, pts_loss])
        if model.use_adaptive_weights:
            # Apply learning rate annealing
            model.loss_weights = model.adaptive_weights(model, losses, tape_annealing)

        # Calculate total loss
        total_loss = reg_loss + res_loss + model.loss_weights * losses

    # Update parameters
    model.optimizer.minimize(total_loss, model.trainable_variables, tape=tape)

    # Update metrics
    model.compiled_metrics.update_state(y_pts, y_pts_pred)

    # Collect metrics to return
    return_metrics = {}
    return_metrics['reg_loss'] = reg_loss
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
    else:
        return_metrics['pts_loss'] = pts_loss

    return return_metrics


def test_step(model, data):
    """Custom test step."""

    data = data_adapter.expand_1d(data)
    x, y, ohe = data_adapter.unpack_x_y_sample_weight(data)
    
    data_res_val = data_adapter.expand_1d(data)

    x_ics, y_ics    = get_next_batch(model.data_ics_val)
    x_pts, y_pts    = get_next_batch(model.data_pts_val)
    x_res, y_res, _ = data_adapter.unpack_x_y_sample_weight(data_res_val)

    # Loss from ICs pts
    y_ics_pred = model(x_ics, training=False)
    ics_loss   = model.ics_loss_fn(y_ics, y_ics_pred)
    # Loss from anchor pts
    y_pts_pred = model(x_pts, training=False)
    pts_loss   = model.compiled_loss(y_pts, y_pts_pred)
    # Loss from residual collocation pts
    y_res_pred, res_loss = model.res_loss_fn(model, x_res, training=False)
    # Regularization Losses
    reg_loss   = losses_utils.get_reg_losses(model)

    # Updates stateful loss metrics.
    model.compiled_metrics.update_state(y_pts, y_pts_pred)
    
    # Collect metrics to return
    return_metrics = {}
    return_metrics['reg_loss'] = reg_loss
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
    else:
        return_metrics['pts_loss'] = pts_loss

    return return_metrics


