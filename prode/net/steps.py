import tensorflow as tf
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter


def train_step(model, data):
    """Custom train step."""

    data = data_adapter.expand_1d(data)
    x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

    with backprop.GradientTape() as tape:
        y_pred, res_loss = model.res_loss_fn(model, x)
        loss = model.compiled_loss(y, y_pred, sample_weight, \
               regularization_losses=model.losses)
        total_loss = loss + res_loss

    model.optimizer.minimize(total_loss, \
               model.trainable_variables, tape=tape)
    model.compiled_metrics.update_state(y, y_pred, sample_weight)

    # Collect metrics to return
    # metr     = {m.name: m.result() for m in model.metrics}
    # res_metr = {'res_loss': res_loss}
    # return {**res_metr, **metr}

    # Collect metrics to return
    return_metrics = {}
    return_metrics['res_loss'] = res_loss
    for metric in model.metrics:
        result = metric.result()
        if isinstance(result, dict):
            return_metrics.update(result)
        else:
            return_metrics[metric.name] = result
    return return_metrics


def test_step(model, data):
    """Custom test step."""
    
    data = data_adapter.expand_1d(data)
    x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

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
