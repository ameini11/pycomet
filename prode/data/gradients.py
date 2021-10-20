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
    grads = [ model(x, training=training) ]
    return grads

def order_n(ode_cls, order, model, ic, t, training):
    if order == 1:
        grad_fn = order_0
    else:
        grad_fn = order_n
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t)
        grads = grad_fn(ode_cls, order-1, model, ic, t, training)
        y = _split(grads[-1])
    dy = _eval_concat(y, t, tape)
    grads.append(dy)
    return grads

# Gradient arrays ops
###############################################################################
def _split(grad):
    fn = lambda x: tf.split(x, num_or_size_splits=x.shape[1], axis=1)
    return [ fn(g) for g in grad ] if isinstance(grad, (list,tuple)) \
        else fn(grad)

def _eval_concat(grad, t, tape):
    fn = lambda x: tf.concat([tape.gradient(i, t) for i in x], axis=1)
    return [ fn(g) for g in grad ] if any(isinstance(g, (list,tuple)) \
        for g in grad) else fn(grad)
