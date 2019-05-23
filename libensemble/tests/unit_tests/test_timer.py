#!/usr/bin/env python

"""
Unit test of timers for libensemble.
"""

import time
from libensemble.util.timer import Timer


def test_timer():
    "Test timer."

    timer = Timer()

    with timer:
        time.sleep(0.5)
        e1 = timer.elapsed

    e2 = timer.elapsed
    time.sleep(0.1)
    e3 = timer.elapsed

    assert (e1 >= 0.5) and (e1 <= 0.6), "Check timed sleep seems correct"
    assert e2 >= e1, "Check timer order."
    assert e2 == e3, "Check elapsed time stable when timer inactive."

    s1 = timer.date_start
    s2 = timer.date_end
    assert s1[0:2] == "20", "Start year is 20xx"
    assert s2[0:2] == "20", "End year is 20xx"

    s3 = "{}".format(timer)
    assert s3 == "Time: {0:.2f} Start: {1} End: {2}".format(e3, s1, s2), \
        "Check string formatting."

    with timer:
        time.sleep(0.5)
        total1 = timer.total

    assert total1 >= 1 and total1 <= 1.1, \
        "Check cumulative timing (active)."
    assert timer.total >= 1 and timer.total <= 1.1, \
        "Check cumulative timing (not active)."
