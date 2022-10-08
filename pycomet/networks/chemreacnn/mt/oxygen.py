import types
import numpy as np
import tensorflow as tf

from keras import layers as tf_layers
from pycomet.networks.basenet import BaseNet
from pycomet import config as cfg
from pycomet import networks
from pycomet import utils

from hypernet.src.general import utils as hyp_utils
thermophysics = hyp_utils.get_thermophysics_mod()
sp_th_mod = thermophysics.specie_thermo
mix_mod = thermophysics.reaction_thermo.mixture


@tf.keras.utils.register_keras_serializable(
    package=cfg._NAME, name='MTOxygenNN'
)
class MTOxygenNN(MMOpNN):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        dim,
        lays_name=None,
        lays_propty=None,
        *args,
        **kwargs
    ):
        # Default layers
        default_lays_propty = {
            'input': {
                'branch': {
                    'index': 0,
                    'xform': None
                },
                'trunk': {
                    'index': 1,
                    'xform': None
                }
            }
        }
        # Layers names
        lays_name = ('input', 'opnns', 'thermo_out')
        not_opt_lays = ('opnns', 'thermo_out')
        super(MTOxygenNN, self).__init__(
            dim=dim,
            lays_name=lays_name,
            lays_propty=lays_propty,
            default_lays_propty=default_lays_propty,
            not_opt_lays=not_opt_lays,
            *args,
            **kwargs
        )

    # Layers initialization
    ###########################################################################
    def init_layers(self):
        lays = {}
        for identifier, propty in self.lays_propty.items():
            if identifier == 'input':
                lays[identifier] = self.init_input_lay(identifier, propty)
            elif identifier == 'opnns':
                lays[identifier] = self.init_opnns_lay(identifier, propty)
            elif identifier == 'thermo_out':
                lays[identifier] = self.init_thermo_out_lay(identifier, propty)
        return lays

    def init_thermo_out_lay(self, identifier, propty):
        _propty = {}
        for k in ('xform',):
            if k in propty and propty[k] is not None:
                if k == 'xform' :
                    if not isinstance(propty[k], tf_layers.Lambda):
                        args = utils.get_fun_args(propty[k])
                        fn = tf_layers.Lambda(
                            function=utils.lambda_lay_dec(propty[k])
                        )
                    else:
                        args = utils.get_fun_args(propty[k].function)
                        propty[k].function = utils.lambda_lay_dec(
                            propty[k].function
                        )
                        fn = propty[k]
                    _propty[k] = {'fn': fn, 'args': args}
                    if any([k not in ('x', 'y') for k in _propty[k]['args']]):
                        utils.raise_value_err(
                            "The outputs transformation function, 'xform', "
                            "accept only ('x', 'y') as function signature, "
                            "where 'x' <- inputs and 'y' <- model ouputs."
                        )
                else:
                    _propty[k] = propty[k]
            else:
                _propty[k] = None
        return _propty

    # Calling
    ###########################################################################
    def _manipulate_output(self, x, y, training):
        if self.active_lays['output']['xform'] is not None:
            lay = self.active_lays['output']['xform']
            inputs = {'x': x, 'y': y}
            y = lay['fn'](
                inputs={k: inputs[k] for k in lay['args']},
                training=training
            )
        return y


@tf.keras.utils.register_keras_serializable(
    package=cfg._NAME, name='MTThermoChemLay'
)
class MTThermoChemLay(tf_layers.Layer):

    def __init__(
        self,
        process,
        specie_thermo,
        out_indices={'Y': {0: [0,1]}, 'T': {1: [0,1]}}
        inp_indices={'e': {0: -1}}
        **kwargs
    ):
        super(ThermoChemLay, self).__init__(
            **kwargs
        )
        self.process = process
        self.specie_thermo = specie_thermo
        self.out_indices = out_indices
        self.inp_indices = inp_indices
        # Specie thermodynamics
        self.sp_th = {
            sp_th['specie']['name']: sp_th_mod.specie_thermo.SpecieThermo(
                index=index,
                process=self.process,
                **sp_th
            ) for index, sp_th in self.specie_thermo.items()
        }
        for name, s in self.sp_th.items():
            if s.specie.n_at > 1:
                self.molecule = name
            else:
                self.atom = name
        # Mixture
        self.mix = utils.get_class(mix_mod, 'MultiComponent')(self.sp_th)

    def call(self, inputs, training=None):
        x, y = inputs
        # Mass fractions and normalized energies (from model outputs)
        Y, norm_e = [
            ops.extract_data(y, self.out_indices[k], axis=1) for k in ('Y','T')
        ]
        # Total energy (from model inputs)
        e = ops.extract_data(x, self.inp_indices['e'], axis=1)
        # Update thermodynamics
        for s in self.sp_th.values():
            s.update(T)
        self.mix.update(Y, var='Y')
        # Formation energy
        e_f = self.mix._e_f()
        # Roto-translational & Vibrational energy
        e_rt, e_v = tf.split(norm_e*(e-e_f), 2, axis=1)
        # Vibrational temperature
        e_v = e_v / self.sp_th[self.molecule].specie.Y
        Tv = self.sp_th[self.molecule].thermo.th_int.ho._T(e_v)
        # Roto-translational temperature
        cv_rt = tf.zeros_like(e)
        for k, s in self.sp_th.items():
            cv_rt = cv_rt + s.thermo.cv_tr*s.specie.Y
            if k == self.molecule:
                cv_rt = cv_rt + s.thermo.th_int.rr.cv*s.specie.Y
        Th = e_rt/cv_rt
        # Return outputs
        return Y, tf.concat([Th, Tv], 1)

    def get_config(self):
        config = {
            "process": self.process,
            "specie_thermo": self.specie_thermo,
            "out_indices": self.out_indices,
            "inp_indices": self.inp_indices
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))