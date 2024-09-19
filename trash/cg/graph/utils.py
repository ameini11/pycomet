from pycomet import utils



def process_layers(gno_cls, lays_name):
  # Layers names
  if (lays_name == None):
    gno_cls.lays_name = gno_cls.lays_name + (
      "fwd_edge_encoder", "node_pf", "edge_bwd", "lin_me_gnet", "residual"
    )
  else:
    gno_cls.lays_name = lays_name
  # Layers properties
  gno_cls.lays_propty = gno_cls.check_layers(
    layers=gno_cls.lays_propty,
    not_opt=(
      "fwd_edge_encoder", "node_pf", "edge_bwd", "lin_me_gnet", "residual"
    ),
    default=gno_cls.default_lays_propty
  )
  # Initializing layers
  gno_cls.active_lays = gno_cls.init_layers()
  gno_cls.init_preinput_lay()



def init_lay(gno_cls, identifier=None, propty=None, lay_id=None):
  check_lay_id(self.name, identifier, propty, lay_id)
  return self.get_lay(propty=propty)

def check_lay_id(gno_name=None, identifier=None, propty=None, lay_id=None):
  if ((lay_id is not None) and (propty["lay_id"] != lay_id)):
    utils.raise_err(
      ValueError,
      f"The layer identifier '{identifier}' in '{gno_name}' " \
      f"must be associated with the '{lay_id}' layer."
    )
