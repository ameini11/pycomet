import tensorflow as tf


def get_grad_fn(ode_order):
    return {
        0: zero,
        1: first,
        2: second
    }[ode_order]


@tf.function
def zero(input_is_list, model, ic, t):
    x = [ic, t]
    if not input_is_list:
        x = tf.concat(x, axis=1)
    y = model(x, training=True)
    return [y]

@tf.function
def first(input_is_list, model, ic, t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t)
        grad  = zero(input_is_list, model, ic, t)
        y     = tf.split(grad[0], num_or_size_splits=grad[0].shape[1], axis=1)
    dy_dt = tf.concat([tape.gradient(y_i, t) for y_i in y], axis=1)
    grad.append(dy_dt)
    return grad

@tf.function
def second(input_is_list, model, ic, t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t)
        grad  = first(input_is_list, model, ic, t)
        dy_dt = tf.split(grad[1], num_or_size_splits=grad[1].shape[1], axis=1)
    d2y_dt2 = tf.concat([tape.gradient(dy_i, t) for dy_i in dy_dt], axis=1)
    grad.append(d2y_dt2)
    return grad
