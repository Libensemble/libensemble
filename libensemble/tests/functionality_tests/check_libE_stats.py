"""Script to check format of libE_stats.txt

Checks matching start and end times existing for calculation and tasks if
required. Checks that dates/times are in a valid format.

"""

from dateutil.parser import parse

infile = "libE_stats.txt"


def is_date(string, fuzzy=False):
    """Return whether the string can be interpreted as a date"""
    try:
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False


def check_datetime(t1, t2):
    """Check if entry is a valid datetime

    Allows for either date and time, or just time
    If just time, then t2 will be non-datetime.
    If second is datetime (time), then merger should be a datetime
    """
    if is_date(t2):
        dt = t1 + " " + t2
    else:
        dt = t1
    assert is_date(dt), f"Expected a datetime, found {dt}"


def check_start_end_times(start="Start:", end="End:", everyline=True):
    """Iterate over rows in infile and check delimiters and datetime formats"""
    with open(infile) as f:
        total_cnt = 0
        for line in f:
            s_cnt = 0
            e_cnt = 0
            lst = line.split()
            if line.startswith("Manager     : Starting") or line.startswith("Manager     : Exiting"):
                check_datetime(lst[5], lst[6])
                continue
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
            assert s_cnt == e_cnt, f"Start/end count different {s_cnt} {e_cnt}"
            total_cnt += s_cnt
        assert total_cnt > 0, f"No timings found starting {start}"


def check_libE_stats(task_datetime=False):
    """Determine and run checks"""
    check_start_end_times()
    if task_datetime:
        check_start_end_times(start="Tstart:", end="Tend:", everyline=False)


if __name__ == "__main__":
    check_libE_stats()
    check_libE_stats(task_datetime=True)
