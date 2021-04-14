import numpy as np
import tensorflow as tf

from .data import Data
from .. import config
from ..utils import run_if_any_none


class FuncConstraint(Data):
    """Function approximation with constraints."""

    def __init__(
        self,
        domain,
        constraint,
        func,
        num_train;
        anchors,
        num_test,
        distr_train="uniform"
    ):
        self.domain      = domain
        self.constraint  = constraint
        self.func        = func
        self.num_train   = num_train
        self.anchors     = anchors
        self.num_test    = num_test
        self.distr_train = distr_train

        self.train_x, self.train_y = None, None
        self.test_x,  self.test_y  = None, None

    def losses(self, targets, loss, net):
        self.train_next_batch()
        self.test()

        n = 0
        if self.anchors is not None:
            n += len(self.anchors)

        f = tf.cond(
            tf.equal(net.data_id, 0),
            lambda: self.constraint(net.inputs, net.outputs, self.train_x),
            lambda: self.constraint(net.inputs, net.outputs, self.test_x),
        )
        return [
            loss(targets[:n], net.outputs[:n]),
            loss(tf.zeros(tf.shape(f), dtype=config.real(tf)), f),
        ]

    @run_if_any_none("train_x", "train_y")
    def get_batch_data(self, batch_size=None):
        if self.distr_train == "log uniform":
            self.train_x = self.domain.log_uniform_points(self.num_train, False)
        elif self.distr_train == "random":
            self.train_x = self.domain.random_points(self.num_train, "sobol")
        else:
            self.train_x = self.domain.uniform_points(self.num_train, False)
        if self.anchors is not None:
            self.train_x = np.vstack((self.anchors, self.train_x))
        self.train_y = self.func(self.train_x)
        return self.train_x, self.train_y

    @run_if_any_none("test_x", "test_y")
    def test(self):
        self.test_x = self.domain.uniform_points(self.num_test, True)
        self.test_y = self.func(self.test_x)
        return self.test_x, self.test_y
        