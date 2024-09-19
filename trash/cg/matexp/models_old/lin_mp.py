import tensorflow as tf
from .blocks import *
from pycomet import utils
from pycomet.networks.basenet import BaseNet


class LinMultiPoleME(BaseNet):
  """Master equation integral solution using multi-pole matrix exponential
  neural operator with a state-specific corrective factor."""

  # Initialization
  # ===================================
  def __init__(
    self,
    dim,
    groups_graph,
    lays_name=None,
    lays_propty=None,
    noisy_io=None,
    inference=None,
    masking=False,
    force_siso=False,
    *args,
    **kwargs
  ):
    super(LinMultiPoleME, self).__init__(
      dim=dim,
      lays_name=lays_name,
      lays_propty=lays_propty,
      noisy_io=noisy_io,
      inference=inference,
      masking=masking,
      force_siso=force_siso,
      *args,
      **kwargs
    )
    # Groups connectivity
    self.grp_conn = GroupsConnectivity(groups_graph)
    # Layers names
    default_lays_name = ("fcorr_xform", "pf_update", "timescales")
    if (lays_name is None):
      self.lays_name = self.lays_name + default_lays_name
    else:
      self.lays_name = lays_name
    # Layers properties
    self.lays_propty = self.check_layers(
      layers=self.lays_propty,
      not_opt=default_lays_name,
      default=self.default_lays_propty
    )
    # Initializing layers
    self.active_lays = self.init_layers()
    self.init_preinput_lay()

  # Layers initialization
  # -----------------------------------
  def init_layers(self):
    lays = {}
    for (identifier, propty) in self.lays_propty.items():
      if (identifier == "fcorr_xform"):
        lays[identifier] = self.init_lay(identifier, propty)
      elif (identifier == "pf_update"):
        propty["lay_id"] = PFUpdate
        lays[identifier] = self.init_lay(identifier, propty)
      elif (identifier == "timescales"):
        lays[identifier] = self.init_timescales_lay(propty)
    lays["temp_update"] = TempUpdate()
    lays["connect"] = CurrentToNext(self.grp_conn.current_to_next["indices"])
    return lays

  def init_timescales_lay(self, propty):
    lays = []
    for (i, ipropty) in enumerate(propty):
      lays.append(
        TimeScale(
          dim=ipropty["dim"],
          lays_propty=ipropty["lays_propty"],
          indices_split=self.grp_conn.current_to_next["conn"][str(i)],
          indices_reduce=self.grp_conn.last_to_current["indices"][str(i+1)]
        )
      )
    return lays

  # Calling
  # ===================================
  def forward(self, inputs, training=None):
    # Extract inputs
    # -------------
    na_0, ni_0, T_0, t = utils.unpack_list(inputs, nb_elem=4)
    # Update thermodynamics
    # -------------
    tlay = self.active_lays["thermo_update"]
    tlay(T, species="molecule", qois="e")
    ei_0 = tlay.sp_th[tlay.species["molecule"]].thermo.e

    nm_0 = tf.math.reduce_sum(ni_0, axis=-1, keepdims=True)
    em_0 = tf.math.reduce_sum(ni_0*ei_0, axis=-1, keepdims=True) / ni_0


    # > t:    time
    # > na:   atom's number density
    # > nm:   molecule's number density
    # > T:    translational temperature
    # > nm_0: molecule's initial distribution function
    # > f:    inputs for state-wise corrective factors



    # Inputs manipulation
    # -------------
    # [y0, t] = inputs
    # >> y0: initial conditions
    # >> t:  time
    y = _manipulate_input(self, inputs)
    # Kernel - Mass / Number density (OpNN)
    # -------------
    n0 = tf.gather(inputs[0], [self.mass_index], axis=-1)
    na = n0 * tf.math.sigmoid(self.active_lays["mass"](y, training=training))
    nm = 0.5*(n0-na)
    # Kernel - Temperature (OpNN)
    # -------------
    T = tf.math.exp(self.active_lays["temp"](y, training=training))
    # Return number densities and temperature
    # -------------
    return na, nm, T





    t, na, nm, T, nm_0, f = utils.unpack_list(inputs, nb_elem=6)
    # Update thermodynamics
    # -------------
    T = self.active_lays["temp_update"](T)
    qa = self.active_lays["pf_update"](T, species="atom")
    # Transform inputs for state-wise latent space
    # -------------
    f = self.active_lays["fcorr_xform"](f, training=training)
    # Loop over timescales
    # -------------
    for lay in self.active_lays["timescales"]:
      nm = lay((t, na, nm, T, qa, nm_0, f))
    return nm

  # def forward(self, inputs, training=None):
  #   # Extract inputs
  #   # -------------
  #   # > t:    time
  #   # > na:   atom's number density
  #   # > nm:   molecule's number density
  #   # > T:    translational temperature
  #   # > nm_0: molecule's initial distribution function
  #   # > f:    inputs for state-wise corrective factors
  #   t, na, nm, T, nm_0, f = utils.unpack_list(inputs, nb_elem=6)
  #   # Update thermodynamics
  #   # -------------
  #   T = self.active_lays["temp_update"](T)
  #   qa = self.active_lays["pf_update"](T, species="atom")
  #   # Transform inputs for state-wise latent space
  #   # -------------
  #   f = self.active_lays["fcorr_xform"](f, training=training)
  #   # Loop over timescales
  #   # -------------
  #   y = [nm]
  #   for lay in self.active_lays["timescales"]:
  #     y.append(lay((t, na, T, qa, nm_0, f)))
  #   # Merge timescales
  #   # -------------
  #   return self.active_lays["connect"](y)
