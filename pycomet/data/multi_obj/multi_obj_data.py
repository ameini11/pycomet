from ..data import Data
from pycomet import utils


class MultiObjData(Data):

  # Initialization
  # ===================================
  def __init__(
    self,
    dim,
    data_src={},
    validation_split=None,
    training=True,
    batch_size=None,
    from_file=False,
    header=0,
    columns=None,
    preproc_fn=None,
    obj_list=[],
    obj_opt_targets=[],
    physics_informed=False
  ):
    super(MultiObjData, self).__init__(
      dim=dim,
      data_src=data_src,
      validation_split=validation_split,
      training=training,
      batch_size=batch_size,
      from_file=from_file,
      header=header,
      columns=columns,
      preproc_fn=preproc_fn
    )
    # Attributes for handling multi-objective loss
    self.multi_obj_loss = True
    # Objectives
    # > Multi-objective functions identifiers (`list`)
    self.obj_list = obj_list
    # > List of objectives with optional targets (meaning having zero as
    #   target value)
    self.obj_opt_targets = obj_opt_targets
    # > Objectives predictive functions (`model.predict` by default)
    self.obj_pred_fn = {k: "model" for k in self.obj_list}
    # > Actual list of objectives identifiers based on data availability
    self.objectives = {
      "train": self.obj_list,
      "valid": self.obj_list
    }
    # Attributes for physics-informed loss
    self.physics_informed = physics_informed

  # Handling Data
  # ===================================
  def get_data_from_src(self, data, identifier):
    if (identifier == "test"):
      return super(MultiObjData, self).get_data_from_src(
        data=data, identifier=identifier
      )
    else:
      dset = {}
      for obj in self.obj_list:
        if (obj in data.keys()):
          kwargs = {}
          if self.from_file:
            for k in ("columns", "preproc_fn"):
              attr = getattr(self, k)
              if ((data[obj] is not None) and (attr is not None)):
                kwargs[k] = attr[identifier][obj]
          dset[obj] = self.extract_data(data=data[obj], **kwargs)
      return dset

  def sample_data(self, identifier="train", max_samples=None):
    dataset = self.train if (identifier == "train") else self.valid
    return dataset.sample_data(max_samples)

  # Logging
  # ===================================
  def get_nb_pts(self, identifier="training", verbose=1):
    def print_fn(dset, identifier, verbose):
      nb_pts = {obj: dset.nb_samples[obj] for obj in dset.data}
      if verbose:
        utils.print_fn(f"Number of {identifier} pts:", lev=1)
        for (k, v) in nb_pts.items():
          utils.print_fn(
            "- '%s':" % k + " "*(5-len(k)) + " %8d" % v, lev=2
          )
    super(MultiObjData, self).get_nb_pts(identifier, verbose, print_fn)
