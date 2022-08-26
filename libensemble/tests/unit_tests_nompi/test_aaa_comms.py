#!/usr/bin/env python

"""
Unit test of termination with comms

Note that this must come before anything that loads PETSc, since PETSc
installs a SIGTERM handler.  Temporarily uninstalled from the test suite,
since pytest slurps up everything (including all the modules) in one go.
"""

import time
import pytest
import signal
import libensemble.comms.comms as comms


def worker_main(comm):
    return


def worker_main_sleeping(comm):
    while True:
        time.sleep(1)


def worker_main_waiting(comm):
    signal.signal(signal.SIGTERM, ignore_handler)
    while not comm.mail_flag():
        pass


def worker_main_sending(comm):
    while not comm.mail_flag():
        comm.send("Hello")
        time.sleep(0.01)


def test_qcomm_proc_terminate1():
    "Test that an already-done QCommProcess gracefully handles terminate()."

    with comms.QCommProcess(worker_main, 2) as mgr_comm:
        time.sleep(0.5)
        mgr_comm.terminate(timeout=30)
        assert not mgr_comm.running


def test_qcomm_proc_terminate2():
    "Test that a QCommProcess run amok can be gracefully terminated."

    with comms.QCommProcess(worker_main_sleeping, 2) as mgr_comm:
        mgr_comm.terminate(timeout=30)
        assert not mgr_comm.running


def ignore_handler(signum, frame):
    print("Ignoring SIGTERM")


@pytest.mark.extra
def test_qcomm_proc_terminate3():
    "Test that a QCommProcess ignoring SIGTERM manages."

    with comms.QCommProcess(worker_main_waiting, 2) as mgr_comm:
        time.sleep(0.5)

        flag = True
        try:
            mgr_comm.recv(timeout=0.5)
            flag = False
        except comms.Timeout:
            pass
        assert flag, "Should time out on recv"

        flag = True
        try:
            mgr_comm.result(timeout=0.5)
            flag = False
        except comms.Timeout:
            pass
        assert flag, "Should time out on result"

        flag = True
        try:
            mgr_comm.terminate(timeout=0.5)
            flag = False
        except comms.Timeout:
            pass
        assert flag, "Should time out on terminate"

        assert mgr_comm.running, "Should still be running"
        mgr_comm.send("Done")


def test_qcomm_proc_terminate4():
    "Test that a QCommProcess can handle event timeouts correctly."

    with comms.QCommProcess(worker_main_sending, 2) as mgr_comm:
        time.sleep(0.5)

        flag = True
        try:
            mgr_comm.result(timeout=0.5)
            flag = False
        except comms.Timeout:
            pass
        assert flag, "Should time out on result"

        assert mgr_comm.running, "Should still be running"
        mgr_comm.send("Done")
