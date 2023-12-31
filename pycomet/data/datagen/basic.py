import abc


class BaseDataGenerator(metaclass=abc.ABCMeta):

  # Initialization
  # ===================================
  def __init__(
    self,
    *args,
    **kwargs
  ):
    pass

  # Source of data
  # ===================================
  @abc.abstractmethod
  def get_data_src(
    self,
    *args,
    **kwargs
  ):
    """Returns data as dictionary with keys: ('train','valid','test')

    Returns:
      A `dict` with keys:
        - 'train' (required)
        - 'valid' (optional)
        - 'test' (optional)
    """
    return {'train': None, 'valid': None, 'test': None}
