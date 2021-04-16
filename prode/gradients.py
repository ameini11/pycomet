import tensorflow as tf


class Jacobian(object):
    """Compute Jacobian matrix J: J[i][j] = dy_i/dx_j, where i=0,...,dim_y-1 and j=0,...,dim_x-1.

    It is lazy evaluation, i.e., it only computes J[i][j] when needed.

    Args:
        ys: Output Tensor of shape (batch_size, dim_y).
        xs: Input Tensor of shape (batch_size, dim_x).
    """

    def __init__(self, outputs_fn, outputs, inputs):
        self.out_fn = outputs_fn

        self.out    = outputs
        self.inp    = inputs

        self.dim_y  = outputs.shape[1]
        self.dim_x  = inputs.shape[1]
        self.J      = {}

    @tf.function
    def __call__(self, i=0, j=0):
        """Returns J[`i`][`j`].
        If `j` is ``None``, returns the gradient of y_i, i.e., J[i].
        """
        if not 0 <= i < self.dim_y:
            raise ValueError("i={} is not valid.".format(i))
        if j is not None and not 0 <= j < self.dim_x:
            raise ValueError("j={} is not valid.".format(j))

        # Compute J[i][j]
        if i not in self.J:
            self.J[i] = {}
        if j not in self.J[i]:
            with tf.GradientTape() as tape:
                x = self.inp[:, j:j+1]
                tape.watch(x)
                y = self.out_fn(self.inp)[:, i:i+1]

            self.J[i][j] = tape.gradient(y, x)

        return self.J[i][j]


class Jacobians(object):
    """Compute multiple Jacobians.

    A new instance will be created for a new pair of (output, input). For the (output, input) pair that has been
    computed before, it will reuse the previous instance, rather than creating a new one.
    """

    def __init__(self):
        self.Js = {}

    def __call__(self, output_fn, outputs, inputs, i=0, j=0):
        key = (outputs.ref(), inputs.ref())              # Put tensors in a set/dictionary.
        if key not in self.Js:
            self.Js[key] = Jacobian(output_fn, outputs, inputs)
        return self.Js[key](i, j)


def jacobian(ys, xs, i=0, j=0):
    """Compute Jacobian matrix J: J[i][j] = dy_i/dx_j, where i=0,...,dim_y-1 and j=0,...,dim_x-1.

    Use this function to compute first-order derivatives instead of ``tf.gradients()``, because

    - It is lazy evaluation, i.e., it only computes J[i][j] when needed.
    - It will remember the gradients that have already been computed to avoid duplicate computation.

    Args:
        ys: Output Tensor of shape (batch_size, dim_y).
        xs: Input Tensor of shape (batch_size, dim_x).
        i (int):
        j (int or None):

    Returns:
        J[`i`][`j`] in Jacobian matrix J. If `j` is ``None``, returns the gradient of y_i, i.e., J[`i`].
    """
    return jacobian._fn(ys, xs, i=i, j=j)

# Initialize the object
jacobian._fn = Jacobians()


# class Hessian(object):
#     """Compute Hessian matrix H: H[i][j] = d^2y/dx_idx_j, where i,j=0,...,dim_x-1.

#     It is lazy evaluation, i.e., it only computes H[i][j] when needed.

#     Args:
#         y: Output Tensor of shape (batch_size, 1) or (batch_size, dim_y > 1).
#         xs: Input Tensor of shape (batch_size, dim_x).
#         component: If `y` has the shape (batch_size, dim_y > 1), then `y[:, component]` is used to compute the
#             Hessian. Do not use if `y` has the shape (batch_size, 1).
#         grad_y: The gradient of `y` w.r.t. `xs`. Provide `grad_y` if known to avoid duplicate computation. `grad_y` can
#             be computed from ``Jacobian``.
#     """

#     def __init__(self, y, xs, component=None, grad_y=None):
#         dim_y = y.get_shape().as_list()[1]
#         if dim_y > 1:
#             if component is None:
#                 raise ValueError("The component of y is missing.")
#             if component >= dim_y:
#                 raise ValueError(
#                     "The component of y={} cannot be larger than the dimension={}.".format(
#                         component, dim_y
#                     )
#                 )
#             y = y[:, component : component + 1]
#         if dim_y == 1 and component is not None:
#             raise ValueError("Do not use component for 1D y.")

#         if grad_y is None:
#             grad_y = tf.gradients(y, xs)[0]
#         self.H = Jacobian(grad_y, xs)

#     def __call__(self, i=0, j=0):
#         """Returns H[`i`][`j`].
#         """
#         return self.H(i, j)


# class Hessians(object):
#     """Compute multiple Hessians.

#     A new instance will be created for a new pair of (output, input). For the (output, input) pair that has been
#     computed before, it will reuse the previous instance, rather than creating a new one.
#     """

#     def __init__(self):
#         self.Hs = {}

#     def __call__(self, y, xs, component=None, i=0, j=0, grad_y=None):
#         _component = 0 if component is None else component
#         key = (y.ref(), xs.ref(), _component)
#         if key not in self.Hs:
#             if grad_y is None:
#                 grad_y = jacobian(y, xs, i=_component, j=None)
#             self.Hs[key] = Hessian(y, xs, component=component, grad_y=grad_y)
#         return self.Hs[key](i, j)


# def hessian(ys, xs, component=None, i=0, j=0, grad_y=None):
#     """Compute Hessian matrix H: H[i][j] = d^2y/dx_idx_j, where i,j=0,...,dim_x-1.

#     Use this function to compute second-order derivatives instead of ``tf.gradients()``, because

#     - It is lazy evaluation, i.e., it only computes H[i][j] when needed.
#     - It will remember the gradients that have already been computed to avoid duplicate computation.

#     Args:
#         ys: Output Tensor of shape (batch_size, dim_y).
#         xs: Input Tensor of shape (batch_size, dim_x).
#         component: If dim_y > 1, then `ys[:, component]` is used as y to compute the Hessian. If dim_y = 1, `component`
#             must be ``None``.
#         i (int):
#         j (int):
#         grad_y: The gradient of y w.r.t. `xs`. Provide `grad_y` if known to avoid duplicate computation. `grad_y` can
#             be computed from ``jacobian``. Even if you do not provide `grad_y`, there is no duplicate computation if
#             you use ``jacobian`` to compute first-order derivatives.

#     Returns:
#         H[`i`][`j`].
#     """
#     return hessian._fn(ys, xs, component=component, i=i, j=j, grad_y=grad_y)


# hessian._fn = Hessians()
