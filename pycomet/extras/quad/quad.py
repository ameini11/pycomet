
def quadrature(func, a, b, args=(), tol=1.49e-8, rtol=1.49e-8, maxiter=50,
               vec_func=True, miniter=1):
    """
    Compute a definite integral using fixed-tolerance Gaussian quadrature.
    Integrate `func` from `a` to `b` using Gaussian quadrature
    with absolute tolerance `tol`.
    Parameters
    ----------
    func : function
        A Python function or method to integrate.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    args : tuple, optional
        Extra arguments to pass to function.
    tol, rtol : float, optional
        Iteration stops when error between last two iterates is less than
        `tol` OR the relative change is less than `rtol`.
    maxiter : int, optional
        Maximum order of Gaussian quadrature.
    vec_func : bool, optional
        True or False if func handles arrays as arguments (is
        a "vector" function). Default is True.
    miniter : int, optional
        Minimum order of Gaussian quadrature.
    Returns
    -------
    val : float
        Gaussian quadrature approximation (within tolerance) to integral.
    err : float
        Difference between last two estimates of the integral.
    See also
    --------
    romberg: adaptive Romberg quadrature
    fixed_quad: fixed-order Gaussian quadrature
    quad: adaptive quadrature using QUADPACK
    dblquad: double integrals
    tplquad: triple integrals
    romb: integrator for sampled data
    simpson: integrator for sampled data
    cumulative_trapezoid: cumulative integration for sampled data
    ode: ODE integrator
    odeint: ODE integrator
    Examples
    --------
    >>> from scipy import integrate
    >>> f = lambda x: x**8
    >>> integrate.quadrature(f, 0.0, 1.0)
    (0.11111111111111106, 4.163336342344337e-17)
    >>> print(1/9.0)  # analytical result
    0.1111111111111111
    >>> integrate.quadrature(np.cos, 0.0, np.pi/2)
    (0.9999999999999536, 3.9611425250996035e-11)
    >>> np.sin(np.pi/2)-np.sin(0)  # analytical result
    1.0
    """
    if not isinstance(args, tuple):
        args = (args,)
    vfunc = vectorize1(func, args, vec_func=vec_func)
    val = np.inf
    err = np.inf
    maxiter = max(miniter+1, maxiter)
    for n in range(miniter, maxiter+1):
        newval = fixed_quad(vfunc, a, b, (), n)[0]
        err = abs(newval-val)
        val = newval

        if err < tol or err < rtol*abs(val):
            break
    else:
        warnings.warn(
            "maxiter (%d) exceeded. Latest difference = %e" % (maxiter, err),
            AccuracyWarning)
    return val, err


def fixed_quad(
    func,
    a,
    b,
    args=(),
    n=5
):
    """
    Compute a definite integral using fixed-order Gaussian quadrature.
    Integrate `func` from `a` to `b` using Gaussian quadrature of
    order `n`.
    Parameters
    ----------
    func : callable
        A Python function or method to integrate (must accept vector inputs).
        If integrating a vector-valued function, the returned array must have
        shape ``(..., len(x))``.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    args : tuple, optional
        Extra arguments to pass to function, if any.
    n : int, optional
        Order of quadrature integration. Default is 5.
    Returns
    -------
    val : float
        Gaussian quadrature approximation to the integral
    none : None
        Statically returned value of None
    See Also
    --------
    quad : adaptive quadrature using QUADPACK
    dblquad : double integrals
    tplquad : triple integrals
    romberg : adaptive Romberg quadrature
    quadrature : adaptive Gaussian quadrature
    romb : integrators for sampled data
    simpson : integrators for sampled data
    cumulative_trapezoid : cumulative integration for sampled data
    ode : ODE integrator
    odeint : ODE integrator
    Examples
    --------
    >>> from scipy import integrate
    >>> f = lambda x: x**8
    >>> integrate.fixed_quad(f, 0.0, 1.0, n=4)
    (0.1110884353741496, None)
    >>> integrate.fixed_quad(f, 0.0, 1.0, n=5)
    (0.11111111111111102, None)
    >>> print(1/9.0)  # analytical result
    0.1111111111111111
    >>> integrate.fixed_quad(np.cos, 0.0, np.pi/2, n=4)
    (0.9999999771971152, None)
    >>> integrate.fixed_quad(np.cos, 0.0, np.pi/2, n=5)
    (1.000000000039565, None)
    >>> np.sin(np.pi/2)-np.sin(0)  # analytical result
    1.0
    """
    x, w = _cached_roots_legendre(n)
    x = tf.math.real(x)
    if np.isinf(a) or np.isinf(b):
        raise ValueError(
            "Gaussian quadrature is only available for finite limits."
            )
    y = (b-a)*(x+1)/2.0 + a
    return (b-a)/2.0 * np.sum(w*func(y, *args), axis=-1), None


def vectorize1(func, args=(), vec_func=False):
    """Vectorize the call to a function.
    This is an internal utility function used by `romberg` and
    `quadrature` to create a vectorized version of a function.
    If `vec_func` is True, the function `func` is assumed to take vector
    arguments.
    Parameters
    ----------
    func : callable
        User defined function.
    args : tuple, optional
        Extra arguments for the function.
    vec_func : bool, optional
        True if the function func takes vector arguments.
    Returns
    -------
    vfunc : callable
        A function that will take a vector argument and return the
        result.
    """
    if vec_func:
        def vfunc(x):
            return func(x, *args)
    else:
        def vfunc(x):
            if np.isscalar(x):
                return func(x, *args)
            x = np.asarray(x)
            # call with first point to get output type
            y0 = func(x[0], *args)
            n = len(x)
            dtype = getattr(y0, 'dtype', type(y0))
            output = np.empty((n,), dtype=dtype)
            output[0] = y0
            for i in range(1, n):
                output[i] = func(x[i], *args)
            return output
    return vfunc



def roots_legendre(n, mu=False):
    r"""Gauss-Legendre quadrature.
    Compute the sample points and weights for Gauss-Legendre
    quadrature. The sample points are the roots of the nth degree
    Legendre polynomial :math:`P_n(x)`. These sample points and
    weights correctly integrate polynomials of degree :math:`2n - 1`
    or less over the interval :math:`[-1, 1]` with weight function
    :math:`w(x) = 1.0`. See 2.2.10 in [AS]_ for more details.
    Parameters
    ----------
    n : int
        quadrature order
    mu : bool, optional
        If True, return the sum of the weights, optional.
    Returns
    -------
    x : ndarray
        Sample points
    w : ndarray
        Weights
    mu : float
        Sum of the weights
    See Also
    --------
    scipy.integrate.quadrature
    scipy.integrate.fixed_quad
    numpy.polynomial.legendre.leggauss
    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.
    """
    m = int(n)
    if n < 1 or n != m:
        raise ValueError("n must be a positive integer.")

    mu0 = 2.0
    an_func = lambda k: 0.0 * k
    bn_func = lambda k: k * np.sqrt(1.0 / (4 * k * k - 1))
    f = cephes.eval_legendre
    df = lambda n, x: (-n*x*cephes.eval_legendre(n, x)
                     + n*cephes.eval_legendre(n-1, x))/(1-x**2)
    return _gen_roots_and_weights(m, mu0, an_func, bn_func, f, df, True, mu)



def _gen_roots_and_weights(n, mu0, an_func, bn_func, f, df, symmetrize, mu):
    """[x,w] = gen_roots_and_weights(n,an_func,sqrt_bn_func,mu)
    Returns the roots (x) of an nth order orthogonal polynomial,
    and weights (w) to use in appropriate Gaussian quadrature with that
    orthogonal polynomial.
    The polynomials have the recurrence relation
          P_n+1(x) = (x - A_n) P_n(x) - B_n P_n-1(x)
    an_func(n)          should return A_n
    sqrt_bn_func(n)     should return sqrt(B_n)
    mu ( = h_0 )        is the integral of the weight over the orthogonal
                        interval
    """
    k = np.arange(n, dtype='d')
    c = np.zeros((2, n))
    c[0,1:] = bn_func(k[1:])
    c[1,:] = an_func(k)
    x = linalg.eigvals_banded(c, overwrite_a_band=True)

    # improve roots by one application of Newton's method
    y = f(n, x)
    dy = df(n, x)
    x -= y/dy

    fm = f(n-1, x)
    fm /= np.abs(fm).max()
    dy /= np.abs(dy).max()
    w = 1.0 / (fm * dy)

    if symmetrize:
        w = (w + w[::-1]) / 2
        x = (x - x[::-1]) / 2

    w *= mu0 / w.sum()

    if mu:
        return x, w, mu0
    else:
        return x, w


# Legendre polynomial
def P(n, x): 
    if(n == 0):
        return 1 # P0 = 1
    elif(n == 1):
        return x # P1 = x
    else:
        return (((2 * n)-1)*x * P(n-1, x)-(n-1)*P(n-2, x))/float(n)