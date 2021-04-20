import abc


class Data(object):
    """Data base class."""

    def __init__(self):
        pass

    @abc.abstractmethod
    def get_batch_data(self, batch_size=None):
        """Return a training dataset of the size `batch_size`."""

    @abc.abstractmethod
    def test(self):
        """Return a test dataset."""
