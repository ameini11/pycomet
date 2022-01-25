import tensorflow as tf

from tensorflow.python.keras import backend
from prode.physics.thermochemistry import specie as specieMdl
from prode.physics.thermochemistry.reactionThermo import mixture as mixMdl
from prode.physics.thermochemistry.chemistry import chemistrySolver as chemSol
from prode.utilities import utils


class PhysicalModel(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        species,
        processFlags,
        mixture='MultiComponent',
        thermo='CoupledEnergyModes',
        constVP='V',
        boxCase='isothermal'
    ):
        # Initilizing thermophysical models -----------------------------------
        # Species thermo
        utils.print_submain("Initializing species thermos ...")
        self.spTh = {
            name: specieMdl.specieThermos.SpecieThermos(
                name,
                rovib,
                thermo=thermo,
                constVP=constVP
            ) for name, rovib in species.items()
        }

        # Mixture
        utils.print_submain("Initializing mixture ...")
        self.mix = utils.get_class(mixMdl, mixture)(self.spTh)

        # Chemistry
        utils.print_submain("Initializing chemistry ...")
        self.chem = chemSol.ChemistrySolver(
            mixture=self.mix,
            specieThermos=self.spTh,
            processFlags=processFlags,
            boxCase='isothermal'
        )

    # ODEs Residual
    ###########################################################################
    def ode(self, ic, t, grads):
        """Evaluates the residual error of a ODEs system given the Initial
        Conditions, the time vector and the n-order derivatives of the model 
        outputs w.r.t. the time input.

        >> Study case herein: Kinetic Master Equations for O2+O system

        Args:
            ic: The Initial Conditions (IC) input `Tensor` with shape 
                (batch_size, d), where `d` is the number of the IC variables.
            t: Time input `Tensor` with shape (batch_size, 1).
            grads: List of `Tensors` (or list of list of `Tensors`) with shape 
                (batch_size, var) of the derivatives of the model outputs
                (`var`) w.r.t. the time input `t`. From 0-th to n-order
                derivatives:
                    d^i(y)/dt^i with i = 0,...,n

        Returns:
            A `Tensor` (or list of `Tensors`) with shape (batch_size, var),
            with `var` the number of variables of the ODEs system.
        """
        # Extract `mass` and `T` vectors from the `ic` tensor
        T, mass = tf.split(tf.gather(ic, [0,1], axis=1), [1,1], axis=1)

        # Extract gradients
        Y, dY = grads

        # Update rates and thermos
        self.chem.update(Y, T, mass)

        # Compute residual (Master Equations)
        res_k = dY*mass - self.chem.dr(Y, mass)

        # Compute mass residual
        res_m = tf.reduce_sum(dY*mass, axis=1, keepdims=True)

        # Compute energy residual
        # res_e = self.chem.dhe(dY)*mass

        return res_k, res_m #, res_e
