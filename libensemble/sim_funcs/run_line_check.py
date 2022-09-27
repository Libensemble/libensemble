from libensemble.message_numbers import WORKER_DONE
from libensemble.executors.executor import Executor
import numpy as np


def circ_offset(wid, length):
    offset = length // 2 + (length % 2 > 0)
    circ_index = wid + offset
    if circ_index > length:
        circ_index = circ_index - length
    return circ_index


def exp_nodelist_for_worker(exp_list, workerID, nodes_per_worker, persis_gens):
    """Modify expected node-lists based on workerID"""
    comps = exp_list.split()
    new_line = []
    for comp in comps:
        if comp.startswith("node-"):
            new_node_list = []
            node_list = comp.split(",")
            for node in node_list:
                node_name, node_num = node.split("-")
                offset = workerID - (1 + persis_gens)
                new_num = int(node_num) + int(nodes_per_worker * offset)
                new_node = "-".join([node_name, str(new_num)])
                new_node_list.append(new_node)
            new_list = ",".join(new_node_list)
            new_line.append(new_list)
        else:
            new_line.append(comp)
    return " ".join(new_line)


def runline_check(H, persis_info, sim_specs, libE_info):
    """Check run-lines produced by executor provided by a list of tests"""
    calc_status = 0
    x = H["x"][0][0]
    exctr = Executor.executor
    test_list = sim_specs["user"]["tests"]
    exp_list = sim_specs["user"]["expect"]
    npw = sim_specs["user"]["nodes_per_worker"]
    p_gens = sim_specs["user"].get("persis_gens", 0)

    for i, test in enumerate(test_list):
        task = exctr.submit(
            calc_type="sim",
            num_procs=test.get("nprocs", None),
            num_nodes=test.get("nnodes", None),
            procs_per_node=test.get("ppn", None),
            extra_args=test.get("e_args", None),
            app_args="--testid " + test.get("testid", None),
            stdout="out.txt",
            stderr="err.txt",
            hyperthreads=test.get("ht", None),
            dry_run=True,
        )

        outline = task.runline
        new_exp_list = exp_nodelist_for_worker(exp_list[i], libE_info["workerID"], npw, p_gens)

        if outline != new_exp_list:
            print(f"outline is: {outline}\nexp     is: {new_exp_list}", flush=True)

        assert outline == new_exp_list

    calc_status = WORKER_DONE
    output = np.zeros(1, dtype=sim_specs["out"])
    output["f"][0] = np.linalg.norm(x)
    return output, persis_info, calc_status


def runline_check_by_worker(H, persis_info, sim_specs, libE_info):
    """Check run-lines produced by executor provided by a list of lines per worker"""
    offset_for_min_rsets_scheduler = sim_specs["user"].get("offset_for_scheduler", False)
    calc_status = 0
    x = H["x"][0][0]
    exctr = Executor.executor
    test = sim_specs["user"]["tests"][0]
    exp_list = sim_specs["user"]["expect"]
    p_gens = sim_specs["user"].get("persis_gens", 0)

    task = exctr.submit(
        calc_type="sim",
        num_procs=test.get("nprocs", None),
        num_nodes=test.get("nnodes", None),
        procs_per_node=test.get("ppn", None),
        extra_args=test.get("e_args", None),
        app_args="--testid " + test.get("testid", None),
        stdout="out.txt",
        stderr="err.txt",
        hyperthreads=test.get("ht", None),
        dry_run=True,
    )

    outline = task.runline
    wid = libE_info["workerID"]

    # Adjust for minimum slot scheduling in alloc func: e.g. 5 rsets on 2 nodes
    # Node 1: 3 rsets (0,1,2). Node 2: 2 rsets (3,4)
    # The first sim will got to rset 3 as it finds a "smaller slot".
    # Alternative would be for splitter to use opposite splits (e.g. 2,3 rather than 3,2)
    if offset_for_min_rsets_scheduler:
        wid_mod = circ_offset(wid, len(exp_list))
    else:
        wid_mod = wid

    new_exp_list = exp_list[wid_mod - 1 - p_gens]

    if outline != new_exp_list:
        print(f"Worker {wid}:\n outline is: {outline}\n exp     is: {new_exp_list}", flush=True)

    assert outline == new_exp_list

    calc_status = WORKER_DONE
    output = np.zeros(1, dtype=sim_specs["out"])
    output["f"][0] = np.linalg.norm(x)
    return output, persis_info, calc_status
