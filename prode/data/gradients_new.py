import tensorflow as tf


def get_grad_fn(ode_order):
    return {
        0: zero,
        1: first,
        2: second
    }[ode_order]


def zero(grads, input_is_list, model, ic, t, training):
    x = [ic, t]
    if not input_is_list:
        x = tf.concat(x, axis=1)
    return grads.write(0, model(x, training=training))

def first(grads, input_is_list, model, ic, t, training):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t)
        grads = zero(grads, input_is_list, model, ic, t, training)
        y     = tf.split(grads.read(0), num_or_size_splits=grads.read(0).shape[1], axis=1)
    dy_dt = tf.concat([tape.gradient(y_i, t) for y_i in y], axis=1)
    return grads.write(1, dy_dt)

def second(grads, input_is_list, model, ic, t, training):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t)
        grads = first(grads, input_is_list, model, ic, t, training)
        dy_dt = tf.split(grads.read(1), num_or_size_splits=grads.read(1).shape[1], axis=1)
    d2y_dt2 = tf.concat([tape.gradient(dy_i, t) for dy_i in dy_dt], axis=1)
    return grads.write(2, d2y_dt2)
