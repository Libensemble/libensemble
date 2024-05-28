from abc import ABC, abstractmethod
import numpy.typing as npt


class LiveData(ABC):

    def __init__(self):
        """Initialize live data capture/processing object"""
        pass

    @abstractmethod
    def live_update(self, hist: npt.NDArray) -> None:
        """Process history data after simulations results have been added to history

        Parameters
        ----------

        hist: :obj:`libensemble.history.History`
            A libEnsemble history type object.
        """

    def finalize(self, hist: npt.NDArray) -> None:
        """
        Finzalize live data processing

        Parameters
        ----------

        hist: :obj:`libensemble.history.History`
            A libEnsemble history type object.
        """
        pass
