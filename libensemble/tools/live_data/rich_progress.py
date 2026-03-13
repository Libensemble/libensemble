"""Optional rich.Progress progress bar for libEnsemble CLI runs."""

try:
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
except ImportError:
    raise ImportError("The 'rich' package is required for RichProgress. Install it with: pip install rich")

from libensemble.tools.live_data.live_data import LiveData


class RichProgress(LiveData):
    """Display a rich progress bar in the terminal during an ensemble run.

    Shows progress toward ``sim_max`` (simulation completions) or
    ``gen_max`` (generator points informed), whichever is set in
    ``exit_criteria``.  If both are set, ``sim_max`` takes priority.

    Parameters
    ----------

    exit_criteria : dict or :class:`libensemble.specs.ExitCriteria`
        The exit criteria used for the run. Must contain either
        ``sim_max`` or ``gen_max`` to show a bounded progress bar.
        If neither is set, an unbounded spinner is displayed instead.

    Examples
    --------

    .. code-block:: python

        from libensemble.tools.live_data.rich_progress import RichProgress

        libE_specs["live_data"] = RichProgress(exit_criteria)
    """

    def __init__(self, exit_criteria=None):
        """Initialize a RichProgress bar.

        Parameters
        ----------
        exit_criteria : dict or ExitCriteria, optional
            Used to determine the total (sim_max or gen_max).
        """
        self._sim_max = None
        self._gen_max = None

        if exit_criteria is not None:
            if hasattr(exit_criteria, "sim_max"):
                # Pydantic model (ExitCriteria)
                self._sim_max = exit_criteria.sim_max
                self._gen_max = exit_criteria.gen_max
            else:
                # Plain dict
                self._sim_max = exit_criteria.get("sim_max")
                self._gen_max = exit_criteria.get("gen_max")

        if self._sim_max is not None:
            total = self._sim_max
            description = "Sims completed"
        elif self._gen_max is not None:
            total = self._gen_max
            description = "Gen points informed"
        else:
            total = None
            description = "Running"

        self._description = description
        self._total = total
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        )
        self._task_id = self._progress.add_task(description, total=total)
        self._progress.start()

    def live_update(self, hist) -> None:
        """Update the progress bar based on the latest history counts."""
        if self._sim_max is not None:
            completed = hist.sim_ended_count
        elif self._gen_max is not None:
            completed = hist.gen_informed_count
        else:
            completed = hist.sim_ended_count

        self._progress.update(self._task_id, completed=completed)

    def finalize(self, hist) -> None:
        """Stop the progress bar display."""
        # Ensure bar reaches 100% if total is known
        if self._total is not None:
            self._progress.update(self._task_id, completed=self._total)
        self._progress.stop()
