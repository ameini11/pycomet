from .data import Data
from pycomet import utils
from pycomet import operations as ops
from tensorflow.python.keras.engine.data_adapter import train_validation_split


class DataSet(Data):

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
    preproc_fn=None
  ):
    super(DataSet, self).__init__(
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
    # Training/Validation data
    if self.training:
      if (("valid" in self.data_src) and (self.data_src["valid"] is not None)):
        for k in ("train", "valid"):
          setattr(self, k, self.get_data_from_src(self.data_src[k], k))
      elif self.validation_split:
        self.train, self.valid = train_validation_split(
          self.get_data_from_src(self.data_src["train"], "train"),
          self.validation_split
        )
      else:
        self.train = self.get_data_from_src(self.data_src["train"], "train")
      self.nb_samples = {
        "train": ops.get_cardinality(self.train),
        "valid": ops.get_cardinality(self.valid)
      }

  # Handling Data
  # ===================================
  def sample_data(self, identifier="train", max_samples=None):
    return utils.sample_dataset(
      dataset=self.train if (identifier == "train") else self.valid,
      nb_samples=self.nb_samples[identifier],
      max_samples=max_samples
    )

  # Logging
  # ===================================
  def get_nb_pts(self, identifier="training", verbose=1):
    def print_fn(dset, identifier, verbose):
      nb_pts = ops.get_cardinality(dset)
      if verbose:
        text = f"Number of {identifier} pts:"
        utils.print_fn(
          text + " "*(30-len(text)) + "%10d" % nb_pts, lev=1
        )
    super(DataSet, self).get_nb_pts(identifier, verbose, print_fn)
