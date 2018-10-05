#!/usr/bin/env python

"""
Unit test of termination with comms

Note that this must come before anything that loads PETSc, since PETSc
installs a SIGTERM handler.
"""

import time
import threading
import queue

import numpy as np
import libensemble.comms as comms


def test_qcomm_proc_terminate():
    "Test that a QCommProcess run amok can be gracefully terminated."

    def worker_main(comm):
        while True:
            time.sleep(1)

    with comms.QCommProcess(worker_main) as mgr_comm:
        mgr_comm.terminate(timeout=30)
        assert not mgr_comm.running
