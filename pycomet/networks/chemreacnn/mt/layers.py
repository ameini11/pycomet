import tensorflow as tf

from tensorflow.python.training.tracking.data_structures import NoDependency
from tensorflow.keras import backend
from keras import layers as tf_layers
from pycomet import operations as ops
from pycomet import config as cfg
from pycomet import utils

from hypernet.src.general import utils as hyp_utils
thermophysics = hyp_utils.get_thermophysics_mod()
mix_mod = thermophysics.reaction_thermo.mixture
sp_th_mod = thermophysics.specie_thermo.specie_thermo


@tf.keras.utils.register_keras_serializable(package=cfg._NAME)
class TwoTempThermoLay(tf_layers.Layer):

    def __init__(
        self,
        process=[],
        specie_thermo={},
        out_indices={'Y': 0, 'T': 1},
        inp_indices={'e': {0: 3}},
        decoding=None,
        T_min=100.,
        **kwargs
    ):
        super(TwoTempThermoLay, self).__init__(**kwargs)
        self.specie_thermo = NoDependency(specie_thermo)
        self.out_indices = out_indices
        self.inp_indices = inp_indices
        self.index_to_key = {i: k for k, i in self.out_indices.items()}
        self.decoding = decoding
        self.T_min = T_min
        # Specie thermodynamics
        self.sp_th = {}
        for index, s in self.specie_thermo.items():
            args = s if 'config' in s \
                else {**s, **dict(index=index, process=process)}
            sp_th = sp_th_mod.SpecieThermo(**args)
            self.sp_th[sp_th.specie.name] = sp_th
        for name, s in self.sp_th.items():
            if s.specie.n_at > 1:
                self.molecule = name
            else:
                self.atom = name
        # Mixture
        self.mix = mix_mod.MultiComponent(self.sp_th)
        
    def call(self, inputs, training=None):
        if self.decoding == 'energy':
            return self.decode_energy(inputs, training)
        elif self.decoding == 'temperature':
            return self.decode_temp(inputs, training)
        elif self.decoding == None:
            return self.encode(inputs, training)
        else:
            utils.raise_value_err("Input 'decoding' not valid.")
        
    def encode(self, inputs, training=None):
        # Recovering inputs/outputs quantities --------------------------------
        x, y = ops.unpack_list(inputs, n_elem=2)
        # Mass fraction (1) and vibrational temperature (1) (from model outputs)
        y = {k: y[self.out_indices[k]] for k in ('Y', 'T')}
        # Enforce mass conservation
        y['Y'] = tf.math.sigmoid(y['Y'])
        y['Y'] = tf.concat([1.-y['Y'], y['Y']],1)
        # Enforce temperature to be positive
        y['T'] = tf.nn.relu(y['T']) + self.T_min
        # Total energy (from model inputs)
        e = ops.extract_data(x, self.inp_indices['e'], axis=1)
        # Recovering energy quantities ----------------------------------------
        # Update thermodynamics
        self.mix.update(y['Y'], var='Y')
        # Formation energy
        e_f = self.mix._e_f()
        # Vibrational energy
        e_v = self.sp_th[self.molecule].thermo.th_int.ho.e(y['T'])
        e_v = e_v * self.sp_th[self.molecule].specie.Y
        # Roto-translational energy
        e_rt = e - e_f - e_v
        # Roto-translational temperature
        cv_rt = tf.zeros_like(e_rt)
        for k, s in self.sp_th.items():
            cv_rt = cv_rt + s.thermo._cv_tr()*s.specie.Y
            if k == self.molecule:
                cv_rt = cv_rt + s.thermo.th_int.rr._cv()*s.specie.Y
        Th = e_rt/cv_rt
        # Return outputs
        y['T'] = tf.concat([Th, y['T']], 1)
        return [y[self.index_to_key[i]] for i in range(2)]
        
    def decode_temp(self, inputs, training=None):
        # Recovering inputs/outputs quantities --------------------------------
        x, y = ops.unpack_list(inputs, n_elem=2)
        # Mass fraction (1) and vibrational temperature (1) (from model outputs)
        y = {k: y[self.out_indices[k]] for k in ('Y', 'T')}
        # Enforce mass conservation
        y['Y'] = tf.math.sigmoid(y['Y'])
        y['Y'] = tf.concat([1.-y['Y'], y['Y']],1)
        # Enforce temperature to be positive
        y['T'] = tf.nn.relu(y['T']) + self.T_min
        # Total energy (from model inputs)
        e = ops.extract_data(x, self.inp_indices['e'], axis=1)
        # Recovering energy quantities ----------------------------------------
        # Update thermodynamics
        self.mix.update(y['Y'], var='Y')
        # Formation energy
        e_f = self.mix._e_f()
        # Vibrational energy
        e_v = self.sp_th[self.molecule].thermo.th_int.ho.e(y['T'])
        e_v = e_v * self.sp_th[self.molecule].specie.Y
        # Roto-translational energy
        e_rt = e - e_f - e_v
        # Roto-translational temperature
        cv_rt = tf.zeros_like(e_rt)
        for k, s in self.sp_th.items():
            cv_rt = cv_rt + s.thermo._cv_tr()*s.specie.Y
            if k == self.molecule:
                cv_rt = cv_rt + s.thermo.th_int.rr._cv()*s.specie.Y
        Th = e_rt/cv_rt
        # Return outputs
        y['T'] = tf.concat([Th, y['T']], 1)
        return [y[self.index_to_key[i]] for i in range(2)]
        
    def decode_energy(self, inputs, training=None):
        x, y = ops.unpack_list(inputs, n_elem=2)
        # Mass fractions and normalized energies (from model outputs)
        y = {k: y[self.out_indices[k]] for k in ('Y', 'T')}
        # Enforce mass/energy conservation
        for k in ('Y', 'T'):
            y[k] = tf.math.sigmoid(y[k])
            y[k] = tf.concat([1.-y[k], y[k]],1)
        # Total energy (from model inputs)
        e = ops.extract_data(x, self.inp_indices['e'], axis=1)
        # Update thermodynamics
        for s in self.sp_th.values():
            s.update(y['T'])
        self.mix.update(y['Y'], var='Y')
        # Formation energy
        e_f = tf.math.minimum(self.mix._e_f(), e*(1.-backend.epsilon()))
        # Roto-translational & Vibrational energy
        e_rt, e_v = tf.split(y['T']*(e-e_f), 2, axis=1)
        # Vibrational temperature
        e_v = e_v/self.sp_th[self.molecule].specie.Y
        Tv = self.sp_th[self.molecule].thermo.th_int.ho._T(e_v)
        # Roto-translational temperature
        cv_rt = tf.zeros_like(e)
        for k, s in self.sp_th.items():
            cv_rt = cv_rt + s.thermo._cv_tr()*s.specie.Y
            if k == self.molecule:
                cv_rt = cv_rt + s.thermo.th_int.rr._cv()*s.specie.Y
        Th = e_rt/cv_rt
        # Return outputs
        y['T'] = tf.concat([Th, Tv], 1)
        return [y[self.index_to_key[i]] for i in range(2)]

    def get_config(self):
        specie_thermo_cfg = {
            sp_th.specie.index: {'config': sp_th.get_config()} \
                for sp_th in self.sp_th.values()
        }
        config = {
            "specie_thermo": specie_thermo_cfg,
            "out_indices": self.out_indices,
            "inp_indices": self.inp_indices,
            "decoding": self.decoding,
            "T_min": self.T_min
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))