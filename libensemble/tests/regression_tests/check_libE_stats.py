import sys
import numpy as np
from dateutil.parser import parse as dtparse

infile = "libE_stats.txt"


def is_date(string, fuzzy=False):
    """
    Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try:
        dtparse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False


def check_datetime(t1, t2):
    """Check if a datetime

    Allows for either date and time, or just time
    If just time, then t2 will be non-datetime.
    If second is datetime (time), then merger should be a datetime
    """
    if is_date(t2):
        dt = t1 + " " + t2
    else:
        dt = t1
    assert is_date(dt), "Expected a datetime, found {}".format(dt)


# Maybe send libE_specs and extract from that
def check_start_end_times(start="Start:", end="End:", everyline=True):
    with open(infile) as f:
        total_cnt = 0
        for line in f:
            s_cnt = 0
            e_cnt = 0
            lst = line.split()
            for i, val in enumerate(lst):
                if val == start:
                    s1 = lst[i + 1]
                    s2 = lst[i + 2]
                    check_datetime(s1, s2)
                    s_cnt += 1
                if val == end:
                    e1 = lst[i + 1]
                    e2 = lst[i + 2]
                    check_datetime(e1, e2)
                    e_cnt += 1
            if everyline:
                assert s_cnt > 0, "Expected timings not found"
            assert s_cnt == e_cnt, "Start/end count different".format(s_cnt, e_cnt)
            total_cnt += s_cnt
        assert total_cnt > 0, "No timings found starting {}".format(start)


def check_libE_stats(task_datetime=False):
    check_start_end_times()
    if task_datetime:
        check_start_end_times(start="Tstart:", end="Tend:", everyline=False)


if __name__ == "__main__":
    check_libE_stats()
    check_libE_stats(task_datetime=True)
