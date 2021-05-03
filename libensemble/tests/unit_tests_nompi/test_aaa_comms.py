#!/usr/bin/env python

"""
Unit test of termination with comms

Note that this must come before anything that loads PETSc, since PETSc
installs a SIGTERM handler.  Temporarily uninstalled from the test suite,
since pytest slurps up everything (including all the modules) in one go.
"""

import time
import signal
from libensemble.tools.tools import osx_set_mp_method
import libensemble.comms.comms as comms

osx_set_mp_method()


def test_qcomm_proc_terminate1():
    "Test that an already-done QCommProcess gracefully handles terminate()."

    def worker_main(comm):
        return

    with comms.QCommProcess(worker_main) as mgr_comm:
        time.sleep(0.5)
        mgr_comm.terminate(timeout=30)
        assert not mgr_comm.running


def test_qcomm_proc_terminate2():
    "Test that a QCommProcess run amok can be gracefully terminated."

    def worker_main(comm):
        while True:
            time.sleep(1)

    with comms.QCommProcess(worker_main) as mgr_comm:
        mgr_comm.terminate(timeout=30)
        assert not mgr_comm.running


def ignore_handler(signum, frame):
    print("Ignoring SIGTERM")


def test_qcomm_proc_terminate3():
    "Test that a QCommProcess ignoring SIGTERM manages."

    def worker_main(comm):
        signal.signal(signal.SIGTERM, ignore_handler)
        while not comm.mail_flag():
            pass

    with comms.QCommProcess(worker_main) as mgr_comm:
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
            mgr_comm.terminate(timeout=1)
            flag = False
        except comms.Timeout:
            pass
        assert flag, "Should time out on terminate"

        assert mgr_comm.running, "Should still be running"
        mgr_comm.send("Done")


def test_qcomm_proc_terminate4():
    "Test that a QCommProcess can handle event timeouts correctly."

    def worker_main(comm):
        while not comm.mail_flag():
            comm.send("Hello")
            time.sleep(0.01)

    with comms.QCommProcess(worker_main) as mgr_comm:
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
