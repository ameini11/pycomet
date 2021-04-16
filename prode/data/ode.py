from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from .data import Data
from .. import config
from ..backend import tf
from ..utils import get_num_args, run_if_all_none


class ODE(Data):
    """ODE solver.

    Args:
        ode:                    A global ODE or a list of ODEs. ``None`` if no global ODE.

        ICs:                    Tensor of ICs: shape=(number_ICs, y_dim)

        domain:                 The number of training residual points sampled inside the domain.
        num_domain (int):       The number of training residual points sampled inside the domain.
        train_distr (string):   The distribution to sample training residual points. One of the following:
                                "uniform" (equispaced grid), "pseudo" (pseudorandom) or "sobol" (Sobol sequence).

        anchors:                A Numpy array of training residual points, in addition to the `num_domain` sampled points.
        exclusions:             A Numpy array of points to be excluded for training.
        solution:               The reference solution.
        num_test:               The number of residual points for testing the ODE residual.
        aux_var_fn:             A function that inputs `train_x` or `test_x` and outputs auxiliary variables.

    Attributes:
        train_x_all:            A Numpy array of all residual points for training. `train_x_all` is unordered,
                                and does not have duplication.
        train_x:                A Numpy array of the residual points fed into the network for training.
                                `train_x` is a subset of `train_x_all`, ordered from BCs to PDE, and may have duplicate points.
        test_x:                 A Numpy array of the residual points fed into the network for testing the PDE residual.
        train_aux_vars:         Auxiliary variables that associate with `train_x`.
        test_aux_vars:          Auxiliary variables that associate with `test_x`.
    """

    # Initialization
    ###########################################################################
    def __init__(
        self,
        ode=None,
        data=None,
        standardize=False,
    ):
        self.ode = ode
        if data is not None:
            self.X, self.Y, self.OHE = data
        else:
            raise ValueError('No data provided.')

        self.scaler_x = None
        if standardize:
            self._standardize()


        # self,
        # ode=None,
        # ode_stability='stable',
        # ICs,
        # domain,
        # num_domain=0,
        # train_distr="sobol",
        # num_anchors=None,
        # num_test=None,
        # aux_var_fn=None,
    
        self.ode         = ode
        self.stability   = ode_stability
        self.ICs         = ICs
        self.n_cases     = np.shape(ICs)[0]
        self.dim         = np.shape(ICs)[1]

        self.domain      = domain
        self.num_domain  = num_domain
        if train_distr not in ["uniform", "pseudo", "sobol"]:
            raise ValueError(
                "train_distr == {}. Available choices: {{'uniform'|'pseudo'|'sobol'}}.".format(
                    train_distr
                )
            )
        self.train_distr = train_distr

        self.num_anchors = num_anchors
        self.num_test    = num_test

        self.train_x, self.train_y = None, None
        self.test_x,  self.test_y  = None, None


    # Losses Definition
    ###########################################################################
    def residual_loss(self, loss, net):
        # Always build the gradients in the PDE here, so that we can reuse all the gradients in dde.grad. If we build
        # the gradients in losses_train(), then error occurs when we use these gradients in losses_test() during
        # sess.run(), because one branch in tf.cond() cannot use the Tensors created in the other branch.
        lss = 0.
        if self.ode is not None:
            F = self.ode(net)
            for f in F:
                targets = tf.zeros(tf.shape(f), dtype=config.real(tf))
                lss    += loss(targets, f)

        return lss

    def ICs_loss(self, targets, loss, net):

        loss(targets, net.outputs)

        self.MSEp = tf.reduce_sum( tf.square( tf.math.multiply(self.encod, tf.math.divide((self.yExac - y), self.yExac)) ) ) / tf.reduce_sum(self.encod)


        f = []
        # Always build the gradients in the PDE here, so that we can reuse all the gradients in dde.grad. If we build
        # the gradients in losses_train(), then error occurs when we use these gradients in losses_test() during
        # sess.run(), because one branch in tf.cond() cannot use the Tensors created in the other branch.
        if self.ode is not None:
            f = self.ode(net)

        if not isinstance(loss, (list, tuple)):
            loss = [loss] * f.shape[1]
        elif len(loss) != f.shape[1]:
            raise ValueError(
                "There are {} ODEs residual, but given {} losses.".format(
                    f.shape[1], len(loss)
                )
            )

        for i, ic in enumerate(self.ics):
            beg, end = ics_start[i], ics_start[i + 1]
            error = ic.error(self.train_x, net.inputs, net.outputs, beg, end)
            losses.append(
                loss[len(error_f) + i](
                    tf.zeros(tf.shape(error), dtype=config.real(tf)), error
                )
            )
        return losses


        return tf.cond(tf.equal(net.data_id, 0), losses_train, losses_test)


    def anchors_loss(self, targets, loss, net):
        f = []
        # Always build the gradients in the PDE here, so that we can reuse all the gradients in dde.grad. If we build
        # the gradients in losses_train(), then error occurs when we use these gradients in losses_test() during
        # sess.run(), because one branch in tf.cond() cannot use the Tensors created in the other branch.
        if self.ode is not None:
            f = self.ode(net)

        if not isinstance(loss, (list, tuple)):
            loss = [loss] * f.shape[1]
        elif len(loss) != f.shape[1]:
            raise ValueError(
                "There are {} ODEs residual, but given {} losses.".format(
                    f.shape[1], len(loss)
                )
            )

        def losses_train():
            f_train = f
            ics_start = np.cumsum([0] + self.num_ics)
            error_f = [fi[ics_start[-1]:] for fi in f_train]
            losses = [
                loss[i](tf.zeros(tf.shape(error), dtype=config.real(tf)), error)
                for i, error in enumerate(error_f)
            ]
            for i, ic in enumerate(self.ics):
                beg, end = ics_start[i], ics_start[i + 1]
                error = ic.error(self.train_x, net.inputs, net.outputs, beg, end)
                losses.append(
                    loss[len(error_f) + i](
                        tf.zeros(tf.shape(error), dtype=config.real(tf)), error
                    )
                )
            return losses

        def losses_test():
            f_test = f
            return [
                loss[i](tf.zeros(tf.shape(fi), dtype=config.real(tf)), fi)
                for i, fi in enumerate(f_test)
            ] + [tf.constant(0, dtype=config.real(tf)) for _ in self.bcs]

        return tf.cond(tf.equal(net.data_id, 0), losses_train, losses_test)


    def losses(self, targets, loss, net):
        l = []
        # Always build the gradients in the PDE here, so that we can reuse all the gradients in dde.grad. If we build
        # the gradients in losses_train(), then error occurs when we use these gradients in losses_test() during
        # sess.run(), because one branch in tf.cond() cannot use the Tensors created in the other branch.
        if self.ode is not None:
            f = self.ode(net)

        if not isinstance(loss, (list, tuple)):
            loss = [loss] * f.shape[1]
        elif len(loss) != f.shape[1]:
            raise ValueError(
                "There are {} ODEs residual, but given {} losses.".format(
                    f.shape[1], len(loss)
                )
            )

        def losses_train():
            f_train = f
            ics_start = np.cumsum([0] + self.num_ics)
            error_f = [fi[ics_start[-1]:] for fi in f_train]
            losses = [
                loss[i](tf.zeros(tf.shape(error), dtype=config.real(tf)), error)
                for i, error in enumerate(error_f)
            ]
            for i, ic in enumerate(self.ics):
                beg, end = ics_start[i], ics_start[i + 1]
                error = ic.error(self.train_x, net.inputs, net.outputs, beg, end)
                losses.append(
                    loss[len(error_f) + i](
                        tf.zeros(tf.shape(error), dtype=config.real(tf)), error
                    )
                )
            return losses

        def losses_test():
            f_test = f
            return [
                loss[i](tf.zeros(tf.shape(fi), dtype=config.real(tf)), fi)
                for i, fi in enumerate(f_test)
            ] + [tf.constant(0, dtype=config.real(tf)) for _ in self.bcs]

        return tf.cond(tf.equal(net.data_id, 0), losses_train, losses_test)
