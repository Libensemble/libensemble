from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt


class LiveData(ABC):
    """A base class for capturing and processing data during an ensemble"""

    def __init__(self):
        """Initialize live data capture/processing object"""
        pass

    @abstractmethod
    def live_update(self, hist: npt.NDArray) -> None:
        """Process history data after simulation results have been added to history

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
