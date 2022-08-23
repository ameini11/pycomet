import numpy as np


class Welford(object):
    
    """ Implements Welford's algorithm for computing a running mean
    and standard deviation as described at: 
        http://www.johndcook.com/standard_deviation.html

    Properties:
        mean    - returns the mean
        std     - returns the std
    """

    def __init__(self):
        self.count = 0
        self._mean = 0.
        self._var = 0.

    def update(self, x=None):
        if x is None:
            return None
        self.count = self.count + 1
        _mean = self._mean + (x - self._mean)/self.count
        _var = self._var + (x - self._mean)*(x - _mean)
        self._mean, self._var = _mean, _var
            
    @property
    def mean(self):
        return self._mean
    
    @property
    def std(self):
        if self.count == 1:
            return 0.
        return np.sqrt(self._var/(self.count-1))