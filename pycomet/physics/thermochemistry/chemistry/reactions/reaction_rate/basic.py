import abc


class Basic(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        reactions_dtb,
        *args,
        **kwargs
    ):
        self.dtb = reactions_dtb

    # Methods
    ###########################################################################
    # Forward reaction rates --------------------------------------------------
    @abc.abstractmethod
    def k_(self, *args, **kwargs):
        pass