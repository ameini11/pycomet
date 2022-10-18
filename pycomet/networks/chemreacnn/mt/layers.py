import tensorflow as tf

from keras import layers as tf_layers
from pycomet import config as cfg
from pycomet import utils

from hypernet.src.general import utils as hyp_utils
thermophysics = hyp_utils.get_thermophysics_mod()
from thermophysics.reaction_thermo.mixture import MultiComponent
from thermophysics.specie_thermo.specie_thermo import SpecieThermo


@tf.keras.utils.register_keras_serializable(package=cfg._NAME)
class TwoTempThermoLay(tf_layers.Layer):

    def __init__(
        self,
        process=[],
        specie_thermo={},
        out_indices={'Y': 0, 'T': 1},
        inp_indices={'e': {0: -1}},
        **kwargs
    ):
        super(TwoTempThermoLay, self).__init__(
            **kwargs
        )
        self.specie_thermo = specie_thermo
        self.out_indices = out_indices
        self.inp_indices = inp_indices
        # Specie thermodynamics
        self.sp_th = {}
        for index, s in self.specie_thermo.items():
            args = s if 'config' in s \
                else s.update(dict(index=index, process=process))
            sp_th = SpecieThermo(**args)
            self.sp_th[sp_th.name] = sp_th
        for name, s in self.sp_th.items():
            if s.specie.n_at > 1:
                self.molecule = name
            else:
                self.atom = name
        # Mixture
        self.mix = MultiComponent(self.sp_th)

    def call(self, inputs, training=None):
        x, y = ops.unpack_list(inputs, n_elem=2)
        # Mass fractions and normalized energies (from model outputs)
        y = {k: y[self.out_indices[k]] for k in ('Y', 'T')}
        # Total energy (from model inputs)
        e = ops.extract_data(x, self.inp_indices['e'], axis=1)
        # Update thermodynamics
        for s in self.sp_th.values():
            s.update(y['T'])
        self.mix.update(y['Y'], var='Y')
        # Formation energy
        e_f = self.mix._e_f()
        # Roto-translational & Vibrational energy
        e_rt, e_v = tf.split(e_hat*(e-e_f), 2, axis=1)
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
        y['T'] = tf.concat([Th, Tv], 1)
        y = list(sorted(
            {i: y[k] for k, i in self.out_indices.items()}
        ).values())
        return y

    def get_config(self):
        specie_thermo_cfg = {
            sp_th.specie.index: {'config': self.sp_th.get_config()} \
                for sp_th in self.sp_th.values()
        }
        config = {
            "specie_thermo": specie_thermo_cfg,
            "out_indices": self.out_indices,
            "inp_indices": self.inp_indices
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))