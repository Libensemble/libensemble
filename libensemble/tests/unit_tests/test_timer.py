#!/usr/bin/env python

"""
Unit test of timers for libensemble.
"""

import time
from libensemble.utils.timer import Timer, TaskTimer


def test_timer():
    "Test timer."

    time_start = time.time()

    timer = Timer()

    with timer:
        time.sleep(0.5)
        e1 = timer.elapsed

    e2 = timer.elapsed
    time.sleep(0.1)
    e3 = timer.elapsed
    time_mid = time.time() - time_start

    # Use external wall-clock time for upper limit to allow for system overhead
    # (e.g. virtualization, or sharing machine with other tasks)
    # assert (e1 >= 0.5) and (e1 <= 0.6), "Check timed sleep seems correct"
    assert (e1 >= 0.5) and (e1 < time_mid), "Check timed sleep within boundaries"
    assert e2 >= e1, "Check timer order."
    assert e2 == e3, "Check elapsed time stable when timer inactive."

    s1 = timer.date_start
    s2 = timer.date_end
    assert s1[0:2] == "20", "Start year is 20xx"
    assert s2[0:2] == "20", "End year is 20xx"

    s3 = f"{timer}"
    assert s3 == f"Time: {e3:.3f} Start: {s1} End: {s2}", "Check string formatting."

    time.sleep(0.2)
    time_start = time.time()
    with timer:
        time.sleep(0.5)
        total1 = timer.total

    time_end = time.time() - time_start + time_mid

    assert total1 >= 1 and total1 <= time_end, "Check cumulative timing (active)."
    assert timer.total >= 1 and timer.total <= time_end, "Check cumulative timing (not active)."


def test_TaskTimer():
    print(TaskTimer())


if __name__ == "__main__":
    test_timer()
    test_TaskTimer()
