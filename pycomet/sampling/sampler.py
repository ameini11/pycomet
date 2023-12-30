import abc

from pycomet import utils


class Sampler(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    name=None,
    *args,
    **kwargs
  ):
    super(Sampler, self).__init__(
      *args,
      **kwargs
    )
    self.name = utils.get_class_name(self).lower()
    if (name is not None):
      self.name = '_'.join((self.name, name))

  # Sampling
  # ===================================
  @abc.abstractmethod
  def sample(self, *args, **kwargs):
    pass
