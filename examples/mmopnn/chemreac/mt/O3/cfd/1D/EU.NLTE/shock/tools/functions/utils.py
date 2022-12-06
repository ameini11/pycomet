import numpy as np


# Define function for string formatting of scientific notation
def sci_notation(
    num,
    decimal_digits=1,
    precision=None,
    exponent=None
):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if exponent is None:
        exponent = int(np.floor(np.log10(abs(num))))
    coeff = np.round(num / float(10**exponent), decimal_digits)
    if precision is None:
        precision = decimal_digits
    return r"${0:.{2}f}\times 10^{{{1:d}}}$".format(coeff, exponent, precision)

# Shock capturing
def capture_shock(x, eps=1.e-2):
    idx = (np.absolute((x - x[0,0])/x[0,0]) > eps).argmax(axis=1)
    return idx