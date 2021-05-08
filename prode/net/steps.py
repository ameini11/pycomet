import tensorflow as tf
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter



@tf.function
def train_step(model, data):
    """Custom train step."""

    data = data_adapter.expand_1d(data)
    x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

    with backprop.GradientTape() as tape:
        y_pred, model.res_loss = model.res_loss_fn(model, x)
        loss = model.compiled_loss(y, y_pred, sample_weight, \
               regularization_losses=model.losses)
        total_loss = loss + model.res_loss

    model.optimizer.minimize(total_loss, \
               model.trainable_variables, tape=tape)
    model.compiled_metrics.update_state(y, y_pred, sample_weight)

    metr     = {m.name: m.result() for m in model.metrics}
    res_metr = {'res_loss': model.res_loss}
    return {**res_metr, **metr}


@tf.function
def test_step(model, data):
    """Custom test step."""
    
    data = data_adapter.expand_1d(data)
    x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

    # Updates stateful loss metrics.
    y_pred, model.res_val_loss = model.res_loss_fn(model, x)

    model.compiled_loss(y, y_pred, sample_weight, \
                       regularization_losses=model.losses)
    model.compiled_metrics.update_state(y, y_pred, sample_weight)

    metr     = {m.name: m.result() for m in model.metrics}
    res_metr = {'res_val_loss': model.res_loss}
    return {**res_metr, **metr}


# @tf.function
# def train_step(model, data):
#     """Custom train step."""

#     data = data_adapter.expand_1d(data)
#     x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

#     with backprop.GradientTape() as tape:
#         if model.res_loss_fn is not None:
#             y_pred, model.res_loss = model.res_loss_fn(model, x)
#         else:
#             y_pred = model(x, training=True)
#         loss = model.compiled_loss(y, y_pred, sample_weight, \
#                regularization_losses=model.losses)

#     model.optimizer.minimize(loss+model.res_loss, \
#                model.trainable_variables, tape=tape)
#     model.compiled_metrics.update_state(y, y_pred, sample_weight)

#     metr = {m.name: m.result() for m in model.metrics}
#     if model.res_loss_fn is not None:
#         res_metr = {'res_loss': model.res_loss}
#         metr = {**res_metr, **metr}
#     return metr


# @tf.function
# def test_step(model, data):
#     """Custom test step."""
    
#     data = data_adapter.expand_1d(data)
#     x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

#     # Updates stateful loss metrics.
#     if model.res_loss_fn is not None:
#         y_pred, model.res_val_loss = model.res_loss_fn(model, x)
#     else:
#         y_pred = model(x, training=True)

#     model.compiled_loss(y, y_pred, sample_weight, \
#                        regularization_losses=model.losses)
#     model.compiled_metrics.update_state(y, y_pred, sample_weight)

#     metr = {m.name: m.result() for m in model.metrics}
#     if model.res_loss_fn is not None:
#         res_metr = {'res_val_loss': model.res_val_loss}
#         metr = {**res_metr, **metr}
#     return metr