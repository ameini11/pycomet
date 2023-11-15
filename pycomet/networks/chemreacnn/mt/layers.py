import tensorflow as tf

from pycomet import utils
from pycomet import config as cfg
from pycomet import operations as ops
from pycomet.networks.elements import layers as pcm_layers
from tensorflow.keras import backend
from tensorflow.python.training.tracking.data_structures import NoDependency

# Hypernet (Version v.22.12)
from hypernet.src.general import utils as hyp_utils
thermophysics = hyp_utils.get_thermophysics_mod()
mix_mod = thermophysics.reaction_thermo.mixture
sp_th_mod = thermophysics.specie_thermo.specie_thermo


@tf.keras.utils.register_keras_serializable(package=cfg._NAME)
class TwoTempThermoLay(pcm_layers.Layer):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        process=[],
        specie_thermo={},
        out_indices_todecode=None,
        inp_indices_encoded=None,
        inp_indices_toencode=None,
        inp_tensor_toencode=0,
        coding_type=None,
        T_min=100.0,
        **kwargs
    ):
        super(TwoTempThermoLay, self).__init__(**kwargs)
        # Do not track 'specie_thermo' dictionary
        self.specie_thermo = NoDependency(specie_thermo)
        # Define variable-index pairs for encoding/decoding
        self.out_indices_todecode = out_indices_todecode
        if self.out_indices_todecode == None:
            self.out_indices_todecode = {'Y': 0, 'T': 1}
        self.out_index_to_key = {
            i: k for k, i in self.out_indices_todecode.items()
        }
        self.inp_tensor_toencode = inp_tensor_toencode
        self.inp_indices_encoded = inp_indices_encoded
        if self.inp_indices_encoded == None:
            self.inp_indices_encoded = {
                'Y_m': 0, 'rho': 1, 'Th': 2, 'e': 3
            }
        self.inp_index_to_key = {
            i: k for k, i in self.inp_indices_encoded.items()
        }
        self.inp_indices_toencode = inp_indices_toencode
        if self.inp_indices_toencode == None:
            self.inp_indices_toencode = {
                'rho_a': 0, 'rho_m': 1, 'Th': 2, 'Tv': 3
            }
        # Define type of encoding/decoding type
        self.coding_type = coding_type
        # Temperature limiter
        self.T_min = T_min
        # Specie thermodynamics
        self.sp_th = {}
        for index, s in self.specie_thermo.items():
            if 'config' in s:
                args = s
            else:
                args = {**s, **dict(index=index, process=process)}
            sp_th = sp_th_mod.SpecieThermo(**args)
            self.sp_th[sp_th.specie.name] = sp_th
        for name, s in self.sp_th.items():
            if s.specie.n_at > 1:
                self.molecule = name
            else:
                self.atom = name
        # Mixture
        self.mix = mix_mod.MultiComponent(self.sp_th)
        
    # Calling
    ###########################################################################
    def call(self, inputs, training=None):
        if self.coding_type == 'decode_energy':
            return self.decode_energy(inputs, training)
        elif self.coding_type == 'decode_temperature':
            return self.decode_temperature(inputs, training)
        elif self.coding_type == 'encode':
            return self.encode(inputs)
        else:
            utils.raise_value_err("Input 'coding_type' not valid.")

    def encode(self, inputs):
        # Extracting inputs quantities to encode
        x = {k: tf.gather(inputs[self.inp_tensor_toencode], [index], axis=1) \
            for k, index in self.inp_indices_toencode.items()}
        # Allocate encoded quantities
        y = {k: 0 for k in self.inp_indices_encoded.keys()}
        # Encode mass quantities
        y['rho'] = x['rho_m'] + x['rho_a']
        y['Y_m'] = x['rho_m'] / y['rho']
        # Encode energy quantities
        Y = tf.concat([x['rho_a'], x['rho_m']], 1) / y['rho']
        T = tf.concat([x['Th'], x['Tv']], 1)
        # Update thermodynamics
        for s in self.sp_th.values():
            s.update(T)
        self.mix.update(Y, var='Y')
        # Total energy and roto-translational temperature (heavy particles)
        y['e'] = self.mix._he()
        y['Th'] = x['Th']
        # Compose and return outputs
        outputs = []
        outputs_i = tf.concat(
            [y[self.inp_index_to_key[i]] for i in range(len(y))], 1
        )
        for i in range(len(inputs)):
            if i == self.inp_tensor_toencode:
                outputs.append(outputs_i)
            else:
                outputs.append(inputs[i])
        return outputs
        
    def decode_temperature(self, inputs, training=None):
        # Recovering inputs/outputs quantities --------------------------------
        y, e = self.extract_inout_qoi(inputs)
        # Enforce mass conservation
        y['Y'] = tf.math.sigmoid(y['Y'])
        y['Y'] = tf.concat([1.-y['Y'], y['Y']],1)
        # Enforce temperature to be positive
        y['T'] = tf.nn.relu(y['T']) + self.T_min
        # Recovering energies/temperatures ------------------------------------
        # Update mixture
        self.mix.update(y['Y'], var='Y')
        # Formation energy
        e_f = self.mix._e_f()
        # Vibrational mode
        e_v = self.sp_th[self.molecule].thermo.th_int.ho.e(y['T'])
        e_v = e_v * self.sp_th[self.molecule].specie.Y
        # Roto-translational mode
        cv_rt = self._cv_rt()
        e_rt_min = cv_rt * self.T_min
        e_rt = tf.nn.relu(e - e_f - e_v - e_rt_min) + e_rt_min
        Th = e_rt/cv_rt
        # Return outputs
        y['T'] = tf.concat([Th, y['T']], 1)
        return [y[self.out_index_to_key[i]] for i in range(len(y))]
        
    def decode_energy(self, inputs, training=None):
        # Recovering inputs/outputs quantities --------------------------------
        y, e = self.extract_inout_qoi(inputs)
        # Enforce mass/energy conservation
        for k in ('Y', 'T'):
            y[k] = tf.math.sigmoid(y[k])
            y[k] = tf.concat([1.-y[k], y[k]], 1)
        # Recovering energies/temperatures ------------------------------------
        # Update mixture
        self.mix.update(y['Y'], var='Y')
        # Formation energy
        e_f_max = e * (1. - backend.epsilon())
        e_f = - tf.nn.relu(e_f_max - self.mix._e_f()) + e_f_max
        # Roto-translational & Vibrational energy
        e_rt, e_v = tf.split(y['T']*(e-e_f), 2, axis=1)
        e_v = e_v/self.sp_th[self.molecule].specie.Y
        # Vibrational temperature
        Tv = self.sp_th[self.molecule].thermo.th_int.ho._T(e_v)
        # Roto-translational temperature
        Th = e_rt/self._cv_rt()
        # Return outputs
        y['T'] = tf.concat([Th, Tv], 1)
        return [y[self.out_index_to_key[i]] for i in range(2)]

    def extract_inout_qoi(self, inputs):
        # Separate inputs from outputs
        x, y = ops.unpack_list(inputs, n_elem=2)
        # Extract model outputs QoI
        #  - Mass fraction/-s (1 or 2)
        #  - Vibrational temperature (1) or normalized energies (2)
        y = {k: y[self.out_indices_todecode[k]] for k in ('Y', 'T')}
        # Extract model inputs QoI
        #  - Total energy
        e = tf.gather(
            x[self.inp_tensor_toencode],
            [self.inp_indices_encoded['e']],
            axis=1
        )
        return y, e

    def _cv_rt(self):
        cv_rt = 0.
        for k, s in self.sp_th.items():
            cv_rt = cv_rt + s.thermo._cv_tr()*s.specie.Y
            if k == self.molecule:
                cv_rt = cv_rt + s.thermo.th_int.rr._cv()*s.specie.Y
        return cv_rt

    # Configuration
    ###########################################################################
    def get_config(self):
        specie_thermo_cfg = {
            sp_th.specie.index: {'config': sp_th.get_config()} \
                for sp_th in self.sp_th.values()
        }
        config = {
            "specie_thermo": specie_thermo_cfg,
            "out_indices_todecode": self.out_indices_todecode,
            "inp_indices_encoded": self.inp_indices_encoded,
            "inp_indices_toencode": self.inp_indices_toencode,
            "inp_tensor_toencode": self.inp_tensor_toencode,
            "coding_type": self.coding_type,
            "T_min": self.T_min
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))