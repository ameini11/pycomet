from .data import Data
from ..utils import run_if_any_none


class Func(Data):
    """Function approximation."""

    def __init__(
        self,
        domain,
        func,
        num_train,
        num_test,
        distr_train="uniform",
        online=False
    ):
        self.domain      = domain
        self.func        = func
        self.num_train   = num_train
        self.num_test    = num_test
        self.distr_train = distr_train
        self.online      = online

        self.train_x, self.train_y = None, None
        self.test_x,  self.test_y  = None, None

    def get_batch_data(self, batch_size=None):
        if self.online:
            self.train_x = self.domain.random_points(batch_size, "pseudo")
            self.train_y = self.func(self.train_x)
        elif self.train_x is None:
            if self.distr_train == "uniform":
                self.train_x = self.domain.uniform_points(self.num_train, True)
            else:
                self.train_x = self.domain.random_points(self.num_train, "sobol")
            self.train_y = self.func(self.train_x)
        return self.train_x, self.train_y

    @run_if_any_none("test_x", "test_y")
    def test(self):
        self.test_x = self.domain.uniform_points(self.num_test, True)
        self.test_y = self.func(self.test_x)
        return self.train_x, self.train_y
