import tensorflow as tf


def get_grad_fn(ode_order):
    return {
        0: zero,
        1: first,
        2: second
    }[ode_order]


# Gradient calculations
###############################################################################
def zero(input_is_list, model, ic, t, training):
    x = [ic, t]
    if not input_is_list:
        x = tf.concat(x, axis=1)
    grads = [ model(x, training=training) ]
    return grads

def first(input_is_list, model, ic, t, training):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t)
        grads = zero(input_is_list, model, ic, t, training)
        y = split_grad(grads[-1])
    dy_dt = eval_concat_grad(y, tape)
    grads.append(dy_dt)
    return grads

def second(input_is_list, model, ic, t, training):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t)
        grads = first(input_is_list, model, ic, t, training)
        dy_dt = split_grad(grads[-1])
    d2y_dt2 = eval_concat_grad(dy_dt, tape)
    grads.append(d2y_dt2)
    return grads


# Array operations
###############################################################################
def split_grad(grad):
    if isinstance(grad, (list,tuple)):
        _grad = []
        for g in grad:
            _grad.append(tf.split(g, num_or_size_splits=g.shape[1], axis=1))
    else:
        _grad = tf.split(grad, num_or_size_splits=grad.shape[1], axis=1)
    return _grad

def eval_concat_grad(grad, tape):
    if any(isinstance(g, (list,tuple)) for g in grad):
        _grad = []
        for g in grad:
            _grad.append(tf.concat([tape.gradient(g_i, t) for g_i in g], axis=1))
    else:
        _grad = tf.concat([tape.gradient(g, t) for g in grad], axis=1)
    return _grad