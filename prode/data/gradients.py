import tensorflow as tf


def get_grad_fn(ode_order=0):
    return order_0 if ode_order == 0 else order_n

# Gradient evaluation
###############################################################################
def order_0(ode_cls, order, model, ic, t, training):
    assert order == 0
    if ode_cls.time_idx == 0:
        x = [t, ic]
    else:
        x = [ic, t]
    if not ode_cls.input_is_list:
        x = tf.concat(x, axis=1)
    y = model(x, training=training)
    if isinstance(y, (list,tuple)):
        y = tf.concat(y, axis=1)
    return [y]

def order_n(ode_cls, order, model, ic, t, training):
    if order == 1:
        grad_fn = order_0
    else:
        grad_fn = order_n
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t)
        grads = grad_fn(ode_cls, order-1, model, ic, t, training)
        # Get `n-1`-th order gradients
        y = tf.split(grads[-1], grads[-1].shape[1], axis=1)
    # Evaluate `n`-th order gradients
    dy = tf.concat([tape.gradient(i, t) for i in y], axis=1)
    # Collect all the gradients
    grads.append(dy)
    return grads
