#!/usr/bin/env python

"""
Unit test of launcher helpers for libensemble.
"""

import sys
import pytest
import libensemble.utils.launcher as launcher


def test_form_command():
    "Test the command templating."

    run_specs = {"mpirun": "mpirun", "nproc": 10, "nrank": 5, "mf": None}

    cmd = ["{mpirun}", "-n {nproc}", "-nper {nrank}", "-machinefile {mf}", 'more arguments "ho hum"']
    args = launcher.form_command(cmd, run_specs)
    aref = ["mpirun", "-n", "10", "-nper", "5", "more", "arguments", '"ho hum"']

    assert args == aref, "Command templating test failed."


def xtest_submit():
    "Test simple launch."

    py_exe = sys.executable or "python"

    # Launch infinite loop, pay attention to term
    process = launcher.launch([py_exe, "launch_busy.py"])
    assert not launcher.process_is_stopped(process, 0.1), "Process stopped early."
    launcher.cancel(process, 0.5)

    # Launch infinite loop, ignore term
    process = launcher.launch([py_exe, "launch_busy.py", "1"])
    assert not launcher.process_is_stopped(process, 0.5), "Process stopped early."
    launcher.cancel(process, 0.5)

    # Launch infinite loop, pay attention to term
    process = launcher.launch([py_exe, "launch_busy.py"], start_new_session=True)
    assert not launcher.process_is_stopped(process, 0.1), "Process stopped early."
    launcher.cancel(process, 0.5)

    # Launch infinite loop, ignore term
    process = launcher.launch([py_exe, "launch_busy.py", "1"], start_new_session=True)
    assert not launcher.process_is_stopped(process, 0.5), "Process stopped early."
    launcher.cancel(process, 0.5)

    # Check proper handling of ProcessLookupError
    assert not launcher.killpg(process), "Expected lookup error."
    assert not launcher.terminatepg(process), "Expected lookup error."

    # Launch finite loop, wait for termination
    process = launcher.launch([py_exe, "launch_busy.py", "0", "0.1"])
    assert launcher.process_is_stopped(process, 1.5), "Process should have stopped earlier."

    # Try simple kill
    process = launcher.launch([py_exe, "launch_busy.py", "1"])
    assert not launcher.process_is_stopped(process, 0.5), "Process stopped early."
    launcher.cancel(process, 0)


@pytest.mark.extra
def test_launch32():
    "If we are in Python > 3.2, still check that 3.2 wait func works"
    saved_wait = launcher.wait
    launcher.wait = launcher.wait_py32
    xtest_submit()
    launcher.wait = saved_wait


@pytest.mark.extra
def test_launch33():
    "If we are in Python > 3.2, also check the new-style wait func"
    if launcher.wait == launcher.wait_py33:
        xtest_submit()
