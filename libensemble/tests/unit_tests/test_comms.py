#!/usr/bin/env python

"""
Unit test of comms for libensemble.
"""

import time
import queue
import threading

import numpy as np
import libensemble.comms as comms


def test_qcomm():
    "Test queue-based bidirectional communicator."

    inq = queue.Queue()
    outq = queue.Queue()
    comm = comms.QComm(inq, outq)

    comm.send('a', 1)
    comm.send('b')
    assert outq.get() == ('a', 1) and outq.get() == ('b',) and outq.empty(), \
      "Check send appropriately goes to output queue."

    inq.put(('c', 3))
    inq.put(('d',))
    assert comm.recv() == ('c', 3) and comm.recv() == ('d',) and inq.empty(), \
      "Check recv appropriately comes from input queue."

    flag = True
    try:
        comm.recv(0.1)
        flag = False
    except comms.Timeout:
        pass
    assert flag, "Check comms receive returns Timeout at appropriate time."


def test_missing_handler():
    "Test correct ABC complaint about missing message handler"

    flag = True
    try:
        class TestHandler(comms.GenCommHandler):
            "Dummy GenCommHandler"

            def on_worker(self, nworker):
                return "on_worker", nworker

            def on_queued(self, sim_id):
                return "on_queued", sim_id

            # Missing on_result

            def on_update(self, sim_id, hist):
                return "on_update", sim_id, hist

            def on_killed(self, sim_id):
                return "on_killed", sim_id

            def on_history(self, hist):
                return "on_history", hist

        TestHandler(None)
        flag = False

    except TypeError:
        pass

    assert flag, "Check define-time error on missing handlers"


def test_gen_comm_handler():
    "Test GenCommHandler interface"

    class TestGenComm(comms.GenCommHandler):
        "Dummy GenComm handler"

        def on_worker(self, nworker):
            return "on_worker", nworker

        def on_queued(self, sim_id):
            return "on_queued", sim_id

        def on_result(self, sim_id, hist):
            return "on_result", sim_id, hist

        def on_update(self, sim_id, hist):
            return "on_update", sim_id, hist

        def on_killed(self, sim_id):
            return "on_killed", sim_id

        def on_history(self, hist):
            return "on_history", hist

    inq = queue.Queue()
    outq = queue.Queue()
    comm = comms.QComm(inq, outq)
    gcomm = TestGenComm(comm)

    gcomm.send_request(None)
    gcomm.send_kill(10)
    gcomm.send_get_history(1, 2)
    gcomm.send_subscribe()

    assert outq.get() == ('request', None)
    assert outq.get() == ('kill', 10)
    assert outq.get() == ('get_history', 1, 2)
    assert outq.get() == ('subscribe',)
    assert outq.empty()

    inq.put(('worker', 3))
    inq.put(('queued', 1))
    inq.put(('result', 1, 100))
    inq.put(('update', 1, 50))
    inq.put(('killed', 1))
    inq.put(('history', 100))
    inq.put(('qwerty',))

    assert gcomm.process_message() == ('on_worker', 3)
    assert gcomm.process_message() == ('on_queued', 1)
    assert gcomm.process_message() == ('on_result', 1, 100)
    assert gcomm.process_message() == ('on_update', 1, 50)
    assert gcomm.process_message() == ('on_killed', 1)
    assert gcomm.process_message() == ('on_history', 100)

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


def test_sim_comm_handler():
    "Test SimCommHandler interface"

    class TestSimComm(comms.SimCommHandler):
        "Dummy SimCommHandler"

        def on_request(self, sim_id, histrecs):
            return "on_request", sim_id, histrecs

        def on_kill(self, sim_id):
            return "on_kill", sim_id

    inq = queue.Queue()
    outq = queue.Queue()
    comm = comms.QComm(inq, outq)
    scomm = TestSimComm(comm)

    scomm.send_result(1, None)
    scomm.send_update(1, 10)
    scomm.send_killed(1)

    assert outq.get() == ('result', 1, None)
    assert outq.get() == ('update', 1, 10)
    assert outq.get() == ('killed', 1)
    assert outq.empty()

    inq.put(('request', 1, 100))
    inq.put(('kill', 1))
    inq.put(('qwerty',))

    assert scomm.process_message() == ('on_request', 1, 100)
    assert scomm.process_message() == ('on_kill', 1)

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


def test_comm_eval():
    "Test CommEval and Future interfaces"

    gen_specs = {'out': [('x', float), ('flag', bool)]}

    inq = queue.Queue()
    outq = queue.Queue()
    comm = comms.QComm(inq, outq)
    gcomm = comms.CommEval(comm, gen_specs=gen_specs)

    inq.put(('history', None))
    inq.put(('worker', 3))
    inq.put(('queued', 1))
    O = np.zeros(2, dtype=gen_specs['out'])
    promises = gcomm.request(O)

    assert len(promises) == 2
    assert gcomm.workers == 3
    assert gcomm.sim_started == 2
    assert gcomm.sim_pending == 2
    assert outq.get(timeout=0.1)[0] == 'request'
    assert outq.empty()

    inq.put(('queued', 3))
    promise = gcomm(10.0, True)
    assert outq.get(timeout=0.1)[0] == 'request'
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
    assert outq.get(timeout=0.1) == ('kill', 3)
    assert not promise.cancelled() and not promise.done()
    inq.put(('killed', 3))
    gcomm.process_message(timeout=0.1)
    assert promise.cancelled() and promise.done()
    assert promise.result() is None
    assert gcomm.sim_started == 3
    assert gcomm.sim_pending == 2

    inq.put(('result', 1, 20))
    gcomm.process_message(timeout=0.1)
    assert not promises[0].cancelled() and promises[0].done()
    assert promises[0].result() == 20
    assert gcomm.sim_started == 3
    assert gcomm.sim_pending == 1

    inq.put(('update', 2, 10))
    gcomm.process_message(timeout=0.1)
    assert not promises[1].cancelled() and not promises[1].done()
    assert gcomm.sim_started == 3
    assert gcomm.sim_pending == 1

    inq.put(('result', 2, 15))
    gcomm.process_message(timeout=0.1)
    assert not promises[1].cancelled() and promises[1].done()
    assert gcomm.sim_started == 3
    assert gcomm.sim_pending == 0

    inq.put(('queued', 4))
    promise = gcomm(x=5.0)
    msg_type, recs = outq.get()
    assert msg_type == 'request'
    assert recs['x'][0] == 5.0 and not recs['flag'][0]
    assert gcomm.sim_started == 4 and gcomm.sim_pending == 1

    inq.put(('killed', 4))
    gcomm.process_message(timeout=0.1)
    assert promise.cancelled() and promise.done()
    assert gcomm.sim_started == 4 and gcomm.sim_pending == 0


def test_thread_comm_eval():

    gen_specs = {'out': [('x', float), ('flag', bool)]}
    inq = queue.Queue()
    outq = queue.Queue()
    comm = comms.QComm(inq, outq)
    gcomm = comms.CommEval(comm, gen_specs=gen_specs)
    mgr_comm = comms.QComm(outq, inq)

    def manager_main():
        "Manager logic for testing CommEval timeouts and waits"
        assert mgr_comm.recv()[0] == 'request'
        mgr_comm.send('queued', 0)
        assert mgr_comm.recv()[0] == 'request'
        mgr_comm.send('queued', 1)
        time.sleep(0.2)
        mgr_comm.send('result', 0, 5)
        time.sleep(0.5)
        mgr_comm.send('result', 1, 10)
        mgr_comm.send('queued', 2)
        mgr_comm.send('queued', 3)
        time.sleep(0.5)
        mgr_comm.send('result', 2, 30)

    mgr_thread = threading.Thread(target=manager_main)
    mgr_thread.start()

    try:
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
    finally:
        mgr_thread.join()
