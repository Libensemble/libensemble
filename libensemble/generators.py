from abc import ABC, abstractmethod
from typing import Optional

import numpy.typing as npt


class AskTellGenerator(ABC):
    """
    Pattern of operations:
    0. User initialize the generator in their script, provides object to libEnsemble
    1. Initial ask for points
    2. Send initial points to libEnsemble for evaluation
    while not instructed to cleanup:
        3. Tell results to generator
        4. Ask for subsequent points
        5. Send points to libEnsemble for evaluation. Get results and any cleanup instruction.
    6. Perform final_tell to generator, retrieve final results if any.
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """
        Initialize the Generator object. Constants and class-attributes go here.
        This will be called only once.

        .. code-block:: python

            my_generator = MyGenerator(my_parameter, batch_size=10)
        """
        pass

    @abstractmethod
    def initial_ask(self, *args, **kwargs) -> npt.NDArray:
        """
        The initial set of generated points is often produced differently than subsequent sets.
        This is a separate method to simplify the common pattern of noting internally if a
        specific ask was the first. This will be called only once.
        """
        pass

    @abstractmethod
    def ask(self, *args, **kwargs) -> npt.NDArray:
        """ """
        pass

    @abstractmethod
    def tell(self, Input: npt.NDArray, *args, **kwargs) -> None:
        """ """
        pass

    @abstractmethod
    def final_tell(self, Input: npt.NDArray, *args, **kwargs) -> Optional[npt.NDArray]:
        """ """
        pass
