import abc


class DataGenerator(metaclass=abc.ABCMeta):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        *args,
        **kwargs
    ):
        self.data_src = {'train': None, 'valid': None, 'test': None}

    # Source of data
    ###########################################################################
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
        return self.data_src
