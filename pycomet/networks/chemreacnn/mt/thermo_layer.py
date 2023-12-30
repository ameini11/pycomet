import tensorflow as tf

from pycomet import utils
from pycomet import config as cfg
from pycomet.networks.elements import layers as pcm_layers
from tensorflow.python.training.tracking.data_structures import NoDependency

# Hypernet (Version v.22.12)
from hypernet.src.general import utils as hyp_utils
thermophysics = hyp_utils.get_thermophysics_mod()
mix_mod = thermophysics.reaction_thermo.mixture
sp_th_mod = thermophysics.specie_thermo.specie_thermo


@tf.keras.utils.register_keras_serializable(package=cfg._NAME)
class ThermoLayerMT(pcm_layers.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    process=[],
    specie_thermo={},
    **kwargs
  ):
    super(ThermoLayerMT, self).__init__(**kwargs)
    # Thermodynamics
    # -----------------------
    # Do not track 'specie_thermo' dictionary
    self.specie_thermo = NoDependency(specie_thermo)
    # Specie thermodynamics
    self.sp_th = {}
    for (index, st) in self.specie_thermo.items():
      if ('config' in st):
        args = st
      else:
        args = {**st, **dict(index=index, process=process)}
      sp_th = sp_th_mod.SpecieThermo(**args)
      self.sp_th[sp_th.specie.name] = sp_th
    # Mixture
    self.mix = mix_mod.MultiComponent(self.sp_th)
    # Number of variables
    # >> Species
    self.nb_comp = self.mix.n_comp
    # >> Temperatures: force 2T model
    self.nb_temp = 2
    if (self.nb_temp > 2):
      utils.raise_value_err(
        f"The current version of '{self.name}' " \
        "layer only supports Park's 2T model."
      )

  # Utils
  # ===================================
  def _is_inp_dict_correct(self, idict):
    c1 = isinstance(idict, dict)
    c2 = False
    if c1:
      c2 = all([(k in idict) for k in ("mass", "temp")])
    return (c1 and c2)

  def _flatten_dict_values(self, idict):
    return {k: tf.nest.flatten(v) for (k, v) in idict.items()}

  # Configuration
  # ===================================
  def get_config(self):
    specie_thermo_CFG = {
      st.specie.index: {'config': st.get_config()} \
        for st in self.sp_th.values()
    }
    config = {
      "specie_thermo": specie_thermo_CFG
    }
    base_config = super(ThermoLayerMT, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
