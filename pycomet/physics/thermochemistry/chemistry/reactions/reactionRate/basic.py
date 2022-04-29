import abc


class Basic(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        reactionsDatabase,
        *args,
        **kwargs
    ):
        self.reacDB = reactionsDatabase

    # Methods
    ###########################################################################
    # Forward reaction rates --------------------------------------------------
    @abc.abstractmethod
    def k_(self, *args, **kwargs):
        pass