#!/usr/bin/env python

"""
Unit test of comms for libensemble.
"""

import logging
import queue as tqueue
import time

import numpy as np

import libensemble.comms.comms as comms
import libensemble.comms.logs as commlogs


def test_qcomm():
    "Test queue-based bidirectional communicator."

    inq = tqueue.Queue()
    outq = tqueue.Queue()
    comm = comms.QComm(inq, outq, 2)

    comm.send("a", 1)
    comm.send("b")
    assert (
        outq.get() == ("a", 1) and outq.get() == ("b",) and outq.empty()
    ), "Check send appropriately goes to output tqueue."

    comm.push_to_buffer("b", 0)
    inq.put(("c", 3))
    inq.put(("d",))
    assert (
        comm.recv() == ("b", 0) and comm.recv() == ("c", 3) and comm.recv() == ("d",) and inq.empty()
    ), "Check recv appropriately comes from input tqueue."

    flag = True
    try:
        comm.recv(0.1)
        flag = False
    except comms.Timeout:
        pass
    assert flag, "Check comms receive returns Timeout at appropriate time."

    assert comm.nworkers == 2, "Check number of workers correct"


def worker_thread(comm, gen_specs):
    gcomm = comms.CommEval(comm, gen_specs=gen_specs)
    p1 = gcomm(x=0.5)
    p2 = gcomm(x=1.0)
    x1 = p1.result()
    assert x1 == 5
    assert not p2.done()
    gcomm.wait_all()
    assert p2.done()
    assert p2.result(timeout=0) == 10
    p3 = gcomm(x=3)
    p4 = gcomm(x=4)
    assert not p3.done()
    assert not p4.done()
    gcomm.wait_any()
    assert p3.done()
    return 128


def bad_worker_thread(comm):
    raise BadWorkerException("Bad worker")


class BadWorkerException(Exception):
    pass


def run_qcomm_threadproc_test(ThreadProc):
    "Test CommEval between two threads or processes (allows timeout checks)"

    gen_specs = {"out": [("x", float), ("flag", bool)]}
    results = np.zeros(3, dtype=[("f", float)])
    results["f"] = [5, 10, 30]
    # resultsf = results['f']
    with ThreadProc(worker_thread, nworkers=2, gen_specs=gen_specs) as mgr_comm:
        assert mgr_comm.running
        assert mgr_comm.recv()[0] == "request"
        mgr_comm.send("queued", 0)
        assert mgr_comm.recv()[0] == "request"
        mgr_comm.send("queued", 1)
        time.sleep(0.2)
        assert not mgr_comm.mail_flag()
        mgr_comm.send("result", 0, results[0])
        time.sleep(0.5)
        mgr_comm.send("result", 1, results[1])
        mgr_comm.send("queued", 2)
        mgr_comm.send("queued", 3)
        time.sleep(0.5)
        mgr_comm.send("result", 2, results[2])
        assert mgr_comm.result() == 128

    try:
        bad_worker_okay = True
        with ThreadProc(bad_worker_thread, nworkers=2) as comm:
            flag = True
            try:
                comm.recv(0.1)
                flag = False
            except comms.Timeout:
                pass
            except comms.CommFinishedException:
                pass
            assert flag, "Test receive timeout from worker"

            _ = comm.result()
            bad_worker_okay = False
    except BadWorkerException:
        pass
    except comms.RemoteException as e:
        assert str(e) == "Bad worker"

    assert bad_worker_okay, "Checking bad worker flag"


def worker_main(comm):
    ch = commlogs.CommLogHandler(comm)
    logger = logging.getLogger()
    logger.addHandler(ch)
    logger.setLevel(logging.INFO)
    logger.info("Test message")
    comm.send("Done!")


def test_comm_logging():
    "Test logging from a worker process is handled correctly."

    with comms.QCommProcess(worker_main, nworkers=2) as mgr_comm:
        msg = mgr_comm.recv()
        assert isinstance(msg[0], logging.LogRecord)


if __name__ == "__main__":
    test_qcomm()
    test_comm_logging()
