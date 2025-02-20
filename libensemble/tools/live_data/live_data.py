from abc import ABC, abstractmethod


class LiveData(ABC):
    """A base class for capturing and processing data during an ensemble"""

    def __init__(self):
        """Initialize live data capture/processing object"""
        pass

    @abstractmethod
    def live_update(self, hist: object) -> None:
        """Process history data after simulation results have been added to history

        Parameters
        ----------

        hist: :obj:`libensemble.history.History`
            A libEnsemble history type object.
        """

    def finalize(self, hist: object) -> None:
        """
        Finzalize live data processing

        Parameters
        ----------

        hist: :obj:`libensemble.history.History`
            A libEnsemble history type object.
        """
        pass
