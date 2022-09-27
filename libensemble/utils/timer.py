"""
libensemble utility class -- manages timer
"""

import datetime


# https://stackoverflow.com/questions/5998245/get-current-time-in-milliseconds-in-python
def TimestampMillisec64():
    return int((datetime.datetime.utcnow() - datetime.datetime(1970, 1, 1)).total_seconds() * 1000)


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
        return f"Time: {self.total:.3f} Start: {self.date_start} End: {self.date_end}"

    @property
    def date_start(self):
        """Return a string representing the start datetime."""
        start_time = datetime.datetime.fromtimestamp(self.tstart / 1000)
        return start_time.strftime("%Y-%m-%d %H:%M:%S") + "." + str(self.tstart)[-3:]

    @property
    def date_end(self):
        """Return a string representing the end datetime."""
        end_time = datetime.datetime.fromtimestamp(self.tend / 1000)
        return end_time.strftime("%Y-%m-%d %H:%M:%S") + "." + str(self.tend)[-3:]

    @property
    def elapsed(self):
        """Return time since last start (active) or in most recent interval."""
        etime = self.tend if not self.timing else TimestampMillisec64()
        return (etime - self.tstart) / 1000

    @property
    def total(self):
        """Return the total time since last start."""
        if self.timing:
            return self.tcum / 1000 + self.elapsed  # second term divided above
        return self.tcum / 1000

    def start(self):
        """Start the timer."""
        self.tstart = TimestampMillisec64()
        self.timing = True

    def stop(self):
        """Stop the timer."""
        self.tend = TimestampMillisec64()
        self.timing = False
        self.tcum += self.tend - self.tstart

    def __enter__(self):
        """Enter a timing context."""
        self.start()
        return self

    def __exit__(self, etype, value, traceback):
        """Exit a timing context."""
        self.stop()


class TaskTimer(Timer):
    """Timer class used in executor tasks."""

    def __str__(self):
        """Return a string representation of the timer."""
        return f"{self.total:.3f} Tstart: {self.date_start} Tend: {self.date_end}"

    def summary(self):
        """Return the total time as a string"""
        return f"{self.total:.3f}"
