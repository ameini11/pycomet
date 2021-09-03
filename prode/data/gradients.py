import tensorflow as tf


def get_grad_fn(ode_order=0):
    return order_0 if ode_order == 0 else order_n

# def get_grad_fn(ode_order):
#     return {
#         0: zero,
#         1: first,
#         2: second
#     }[ode_order]


# Gradient evaluation
###############################################################################
# def zero(ode_cls, model, ic, t, training):
#     if ode_cls.time_idx == 0:
#         t = [t, ic]
#     else:
#         t = [ic, t]
#     if not ode_cls.input_is_list:
#         x = tf.concat(x, axis=1)
#     grads = [ model(x, training=training) ]
#     return grads

# def first(ode_cls, model, ic, t, training):
#     with tf.GradientTape(persistent=True) as tape:
#         tape.watch(t)
#         grads = zero(ode_cls, model, ic, t, training)
#         y = split_grad(grads[-1])
#     dy_dt = eval_concat_grad(y, tape)
#     grads.append(dy_dt)
#     return grads

# def second(ode_cls, model, ic, t, training):
#     with tf.GradientTape(persistent=True) as tape:
#         tape.watch(t)
#         grads = first(ode_cls, model, ic, t, training)
#         dy_dt = split_grad(grads[-1])
#     d2y_dt2 = eval_concat_grad(dy_dt, tape)
#     grads.append(d2y_dt2)
#     return grads


def order_0(ode_cls, order, model, ic, t, training):
    assert order == 0
    if ode_cls.time_idx == 0:
        t = [t, ic]
    else:
        t = [ic, t]
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
        y = split_grad(grads[-1])
    dy = eval_concat_grad(y, t, tape)
    grads.append(dy)
    return grads


# Gradient arrays ops
###############################################################################
def split_grad(grad):
    fn = lambda x: tf.split(x, num_or_size_splits=x.shape[1], axis=1)
    return [ fn(g) for g in grad ] if isinstance(grad, (list,tuple)) \
        else fn(grad)

def eval_concat_grad(grad, t, tape):
    fn = lambda x: tf.concat([tape.gradient(i, t) for i in x], axis=1)
    return [ fn(g) for g in grad ] if any(isinstance(g, (list,tuple)) \
        for g in grad) else fn(grad)
