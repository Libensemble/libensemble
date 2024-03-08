from abc import ABC, abstractmethod
from typing import Iterable, Optional


class Generator(ABC):
    """

    Tentative generator interface for use with libEnsemble, and generic enough to be
    broadly compatible with other workflow packages.

    .. code-block:: python

        from libensemble import Ensemble
        from libensemble.generators import Generator


        class MyGenerator(Generator):
            def __init__(self, param):
                self.param = param
                self.model = None

            def initial_ask(self, num_points, yesterdays_points):
                return create_initial_points(num_points, self.param, yesterdays_points)

            def ask(self, num_points):
                return create_points(num_points, self.param)

            def tell(self, results):
                self.model = update_model(results, self.model)

            def final_tell(self, results):
                self.tell(results)
                return list(self.model)


        my_generator = MyGenerator(my_parameter=100)
        my_ensemble = Ensemble(generator=my_generator)

    Pattern of operations:
    0. User initializes the generator class in their script, provides object to workflow/libEnsemble
    1. Initial ask for points from the generator
    2. Send initial points to workflow for evaluation
    while not instructed to cleanup:
        3. Tell results to generator
        4. Ask generator for subsequent points
        5. Send points to workflow for evaluation. Get results and any cleanup instruction.
    6. Perform final_tell to generator, retrieve any final results/points if any.

    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """
        Initialize the Generator object on the user-side. Constants, class-attributes,
        and preparation goes here.

        .. code-block:: python

            my_generator = MyGenerator(my_parameter, batch_size=10)
        """

    def initial_ask(self, num_points: int, previous_results: Optional[Iterable]) -> Iterable:
        """
        The initial set of generated points is often produced differently than subsequent sets.
        This is a separate method to simplify the common pattern of noting internally if a
        specific ask was the first. Previous results can be provided to build a foundation
        for the initial sample. This will be called only once.
        """

    @abstractmethod
    def ask(self, num_points: int) -> Iterable:
        """
        Request the next set of points to evaluate.
        """

    def tell(self, results: Iterable) -> None:
        """
        Send the results of evaluations to the generator.
        """

    def final_tell(self, results: Iterable) -> Optional[Iterable]:
        """
        Send the last set of results to the generator, instruct it to cleanup, and
        optionally retrieve an updated final state of evaluations. This is a separate
        method to simplify the common pattern of noting internally if a
        specific tell is the last. This will be called only once.
        """
