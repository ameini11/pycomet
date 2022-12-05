import numpy as np

from scipy import ndimage


def compute_err_2d(
    y,
    x=None,
    shift=None,
    safe_idx=2,
    wind_size_flt=2,
    lin_flt=False,
    lin_flt_idx=(500, 900)
):
    # Error
    err = ape(y['surrogate'], y['thermolib'])
    # Shock reference frame
    if shift is not None:
        err_ = []
        for i, err_i in enumerate(err):
            idx = max(shift['thermolib'][i], shift['surrogate'][i]) + safe_idx
            err_i[:idx] = 0.
            err_i = np.roll(err_i, -int(safe_idx/2))
            err_.append(err_i)
        err = np.vstack(err_)
    # Linear avarage filter
    if lin_flt:
        p0 = (x[lin_flt_idx[0]], 1.e-1)
        p1 = (x[lin_flt_idx[1]], 1.)
        y_lin = lin_fn(x, p0, p1)
        err_ = []
        sh_0 = shift['thermolib'][0]
        for i, err_i in enumerate(err):
            idx = max(shift['thermolib'][i], shift['surrogate'][i]) + safe_idx
            err_i = np.roll(err_i, -(idx - sh_0))
            err_i = err_i * y_lin
            err_i = np.roll(err_i, (idx - sh_0))
            err_.append(err_i)
        err = np.vstack(err_)
    # Window avarage filter
    err = ndimage.uniform_filter(err, size=wind_size_flt)
    return err

# Error functions
def ape(y_pred, y_true):
    return 100 * np.absolute((y_true - y_pred) / y_true)

def ae(y_pred, y_true):
    return np.absolute(y_true - y_pred)

def eval_err(y_pred, y_true, norm_fn='mean', err_fn='ape'):
    if norm_fn == 'mean':
        norm_fn_ = np.mean
    elif norm_fn == 'max':
        norm_fn_ = np.amax
    if err_fn == 'ape':
        err_fn_ = ape
    elif err_fn == 'ae':
        err_fn_ = ae
    return norm_fn_(err_fn_(y_pred, y_true))

# Shock capturing
def capture_shock(x, eps=1.e-2):
    idx = (np.absolute((x - x[0,0])/x[0,0]) > eps).argmax(axis=1)
    return idx

# Linear avarage filtering
def lin_fn(x, p0, p1):
    a = (p1[1]-p0[1])/(p1[0]-p0[0])
    b = p0[1] - a*p0[0]
    return a*x + b