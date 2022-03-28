import tensorflow as tf

GRAD_OPT = 2


# Gradient functions
###############################################################################
def order_0(time_idx, order, model, ic, t, training):
    # Inputs
    x = [t, ic] if time_idx == 0 else [ic, t]
    if len(tf.nest.flatten(model.inputs)) == 1:
        x = tf.concat(x, axis=1)
    # Outputs
    y = model(x, training=training)
    if len(tf.nest.flatten(model.outputs)) > 1:
        y = tf.concat(y, axis=1)
    return [y]

if GRAD_OPT == 1:
    def order_n(time_idx, order, model, ic, t, training):
        grad_fn = order_0 if order == 1 else order_n
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            grads = grad_fn(time_idx, order-1, model, ic, t, training)
        # Evaluate `n`-th order gradients
        dy = tape.batch_jacobian(grads[-1], t, experimental_use_pfor=True)
        dy = tf.squeeze(dy, axis=-1)
        # Collect `n`-th order gradients
        grads.append(dy)
        return grads

elif GRAD_OPT == 2:
    def order_n(time_idx, order, model, ic, t, training):
        grad_fn = order_0 if order == 1 else order_n
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            grads = grad_fn(time_idx, order-1, model, ic, t, training)
            # Get `n-1`-th order gradients
            y = tf.split(grads[-1], grads[-1].shape[1], axis=1)
        # Evaluate `n`-th order gradients
        dy = tf.concat([tape.gradient(yi, t) for yi in y], axis=1)
        # Collect `n`-th order gradients
        grads.append(dy)
        return grads

# Gradient evaluation
###############################################################################
def eval(time_idx, order, model, ic, t, training):
    grad_fn = order_0 if order == 0 else order_n
    grads = grad_fn(time_idx, order, model, ic, t, training)
    if len(tf.nest.flatten(model.outputs)) > 1:
        grads = [tf.split(g, model.output_dim, axis=1) for g in grads]
    return tuple(grads)
