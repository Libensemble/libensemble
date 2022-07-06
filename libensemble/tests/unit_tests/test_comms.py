#!/usr/bin/env python

"""
Unit test of comms for libensemble.
"""

import time
import queue
import logging

import numpy as np
import libensemble.comms.comms as comms
import libensemble.comms.logs as commlogs


def test_qcomm():
    "Test queue-based bidirectional communicator."

    inq = queue.Queue()
    outq = queue.Queue()
    comm = comms.QComm(inq, outq, 2)

    comm.send("a", 1)
    comm.send("b")
    assert (
        outq.get() == ("a", 1) and outq.get() == ("b",) and outq.empty()
    ), "Check send appropriately goes to output queue."

    comm.push_to_buffer("b", 0)
    inq.put(("c", 3))
    inq.put(("d",))
    assert (
        comm.recv() == ("b", 0) and comm.recv() == ("c", 3) and comm.recv() == ("d",) and inq.empty()
    ), "Check recv appropriately comes from input queue."

    flag = True
    try:
        comm.recv(0.1)
        flag = False
    except comms.Timeout:
        pass
    assert flag, "Check comms receive returns Timeout at appropriate time."

    assert comm.nworkers == 2, "Check number of workers correct"


def test_missing_handler():
    "Test correct ABC complaint about missing message handler"

    flag = True
    try:

        class TestHandler(comms.GenCommHandler):
            "Dummy GenCommHandler"

            def on_worker_avail(self, nworker):
                return "on_worker_avail", nworker

            def on_queued(self, sim_id):
                return "on_queued", sim_id

            # Missing on_result

            def on_update(self, sim_id, recs):
                return "on_update", sim_id, recs

            def on_killed(self, sim_id):
                return "on_killed", sim_id

        TestHandler(None)
        flag = False

    except TypeError:
        pass

    assert flag, "Check define-time error on missing handlers"


def test_gen_comm_handler():
    "Test GenCommHandler interface"

    class TestGenComm(comms.GenCommHandler):
        "Dummy GenComm handler"

        def on_worker_avail(self, nworker):
            return "on_worker_avail", nworker

        def on_queued(self, sim_id):
            return "on_queued", sim_id

        def on_result(self, sim_id, recs):
            return "on_result", sim_id, recs

        def on_update(self, sim_id, recs):
            return "on_update", sim_id, recs

        def on_killed(self, sim_id):
            return "on_killed", sim_id

    inq = queue.Queue()
    outq = queue.Queue()
    comm = comms.QComm(inq, outq, 4)
    gcomm = TestGenComm(comm)

    gcomm.send_request(None)
    gcomm.send_kill(10)
    gcomm.send_get_history(1, 2)
    gcomm.send_subscribe()

    assert outq.get() == ("request", None)
    assert outq.get() == ("kill", 10)
    assert outq.get() == ("get_history", 1, 2)
    assert outq.get() == ("subscribe",)
    assert outq.empty()

    inq.put(("worker_avail", 3))
    inq.put(("queued", 1))
    inq.put(("result", 1, 100))
    inq.put(("update", 1, 50))
    inq.put(("killed", 1))
    inq.put(("qwerty",))

    assert gcomm.process_message() == ("on_worker_avail", 3)
    assert gcomm.process_message() == ("on_queued", 1)
    assert gcomm.process_message() == ("on_result", 1, 100)
    assert gcomm.process_message() == ("on_update", 1, 50)
    assert gcomm.process_message() == ("on_killed", 1)

    flag = True
    try:
        gcomm.process_message(0.1)
    except ValueError:
        pass
    assert flag, "Check error handling for bad message type"

    flag = True
    try:
        gcomm.process_message(0.1)
        flag = False
    except comms.Timeout:
        pass
    assert flag, "Check timeout on process_message"

    flag = True
    try:
        inq.put(("stop",))
        gcomm.process_message(0.1)
        flag = False
    except comms.ManagerStop:
        pass
    assert flag, "Check exception raised on manager requested stop"


def test_sim_comm_handler():
    "Test SimCommHandler interface"

    class TestSimComm(comms.SimCommHandler):
        "Dummy SimCommHandler"

        def on_request(self, sim_id, recs):
            return "on_request", sim_id, recs

        def on_kill(self, sim_id):
            return "on_kill", sim_id

    inq = queue.Queue()
    outq = queue.Queue()
    comm = comms.QComm(inq, outq, 2)
    scomm = TestSimComm(comm)

    scomm.send_result(1, None)
    scomm.send_update(1, 10)
    scomm.send_killed(1)

    assert outq.get() == ("result", 1, None)
    assert outq.get() == ("update", 1, 10)
    assert outq.get() == ("killed", 1)
    assert outq.empty()

    inq.put(("request", 1, 100))
    inq.put(("kill", 1))
    inq.put(("qwerty",))

    assert scomm.process_message() == ("on_request", 1, 100)
    assert scomm.process_message() == ("on_kill", 1)

    flag = True
    try:
        scomm.process_message(0.1)
    except ValueError:
        pass
    assert flag, "Check error handling for bad message type"

    flag = True
    try:
        scomm.process_message(0.1)
        flag = False
    except comms.Timeout:
        pass
    assert flag, "Check timeout on process_message"

    flag = True
    try:
        inq.put(("stop",))
        scomm.process_message(0.1)
        flag = False
    except comms.ManagerStop:
        pass
    assert flag, "Check exception raised on manager requested stop"


def test_comm_eval():
    "Test CommEval and Future interfaces"

    gen_specs = {"out": [("x", float), ("flag", bool)]}

    inq = queue.Queue()
    outq = queue.Queue()
    comm = comms.QComm(inq, outq, 2)
    gcomm = comms.CommEval(comm, gen_specs=gen_specs)

    inq.put(("worker_avail", 3))
    inq.put(("queued", 1))
    H_o = np.zeros(2, dtype=gen_specs["out"])
    promises = gcomm.request(H_o)

    assert len(promises) == 2
    assert gcomm.workers == 3
    assert gcomm.sim_started == 2
    assert gcomm.sim_pending == 2
    assert outq.get(timeout=0.1)[0] == "request"
    assert outq.empty()

    inq.put(("queued", 3))
    promise = gcomm(10.0, True)
    assert outq.get(timeout=0.1)[0] == "request"
    assert not promise.cancelled() and not promise.done()
    assert gcomm.sim_started == 3
    assert gcomm.sim_pending == 3

    flag = True
    try:
        promise.result(timeout=0.1)
        flag = False
    except comms.Timeout:
        pass
    assert flag, "Checking timeout on result check"

    promise.cancel()
    assert outq.get(timeout=0.1) == ("kill", 3)
    assert not promise.cancelled() and not promise.done()
    inq.put(("killed", 3))
    gcomm.process_message(timeout=0.1)
    assert promise.cancelled() and promise.done()
    assert promise.result() is None
    assert gcomm.sim_started == 3
    assert gcomm.sim_pending == 2

    results = np.zeros(3, dtype=[("f", float)])
    results["f"] = [20, 10, 15]
    resultsf = results["f"]

    resultsf[0] = 15
    inq.put(("update", 1, results[0]))
    gcomm.process_message(timeout=0.1)
    assert not promises[0].cancelled() and not promises[0].done()
    assert promises[0].current_result == 15
    assert gcomm.sim_started == 3
    assert gcomm.sim_pending == 2

    resultsf[0] = 20
    inq.put(("result", 1, results[0]))
    gcomm.process_message(timeout=0.1)
    assert not promises[0].cancelled() and promises[0].done()
    assert promises[0].result() == 20
    assert gcomm.sim_started == 3
    assert gcomm.sim_pending == 1

    inq.put(("update", 2, results[1]))
    gcomm.process_message(timeout=0.1)
    assert not promises[1].cancelled() and not promises[1].done()
    assert gcomm.sim_started == 3
    assert gcomm.sim_pending == 1

    inq.put(("result", 2, results[2]))
    gcomm.process_message(timeout=0.1)
    assert not promises[1].cancelled() and promises[1].done()
    assert gcomm.sim_started == 3
    assert gcomm.sim_pending == 0

    inq.put(("queued", 4))
    promise = gcomm(x=5.0)
    msg_type, recs = outq.get()
    assert msg_type == "request"
    assert recs["x"][0] == 5.0 and not recs["flag"][0]
    assert gcomm.sim_started == 4 and gcomm.sim_pending == 1

    inq.put(("killed", 4))
    gcomm.process_message(timeout=0.1)
    assert promise.cancelled() and promise.done()
    assert gcomm.sim_started == 4 and gcomm.sim_pending == 0


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


def test_qcomm_threadproc():
    "Test CommEval between threads and processes"
    run_qcomm_threadproc_test(comms.QCommThread)
    run_qcomm_threadproc_test(comms.QCommProcess)


def test_comm_logging():
    "Test logging from a worker process is handled correctly."

    with comms.QCommProcess(worker_main, nworkers=2) as mgr_comm:
        msg = mgr_comm.recv()
        assert isinstance(msg[0], logging.LogRecord)
