"""
libensemble utility class -- manages timer
"""

import time


class Timer:
    """Timer class used in libensemble.

    Attributes
    ----------

    tcum: float:
        Total time recorded by timer.

    tstart: float:
        Most recent starting time.

    tend: float:
        Most recent ending time.

    timing: bool:
        Indicates whether the timer is currently active.
    """

    def __init__(self):
        """Initialize a new timer."""
        self.tcum = 0.0
        self.tstart = 0.0
        self.tend = 0.0
        self.timing = False

    def __str__(self):
        """Return a string representation of the timer."""
        return ("Time: {0:.2f} Start: {1} End: {2}".
                format(self.total, self.date_start, self.date_end))

    @property
    def date_start(self):
        """Return a string representing the start datetime."""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.tstart))

    @property
    def date_end(self):
        """Return a string representing the end datetime."""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.tend))

    @property
    def elapsed(self):
        """Return time since last start (active) or in most recent interval."""
        etime = self.tend if not self.timing else time.time()
        return etime-self.tstart

    @property
    def total(self):
        """Return the total time since last start."""
        if self.timing:
            return self.tcum + self.elapsed
        return self.tcum

    def start(self):
        """Start the timer."""
        self.tstart = time.time()
        self.timing = True

    def stop(self):
        """Stop the timer."""
        self.tend = time.time()
        self.timing = False
        self.tcum += (self.tend-self.tstart)

    def __enter__(self):
        """Enter a timing context."""
        self.start()
        return self

    def __exit__(self, etype, value, traceback):
        """Exit a timing context."""
        self.stop()


class JobTimer(Timer):
    """Timer class used in job controller jobs."""
    def __str__(self):
        """Return a string representation of the timer."""
        return ("JobTime: {0:.2f} JStart: {1} JEnd: {2}".
                format(self.total, self.date_start, self.date_end))
