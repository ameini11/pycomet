import abc


class Sampler(object):

    # Initialization
    ###########################################################################
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
        self.name = self.__class__.__name__.lower()
        if name is not None:
            self.name = '_'.join((self.name, name))

    # Sampling ----------------------------------------------------------------
    @abc.abstractmethod
    def sample(self, *args, **kwargs):
        pass
