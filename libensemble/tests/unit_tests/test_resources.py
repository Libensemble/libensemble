import os
import socket

import numpy as np

from libensemble.executors.mpi_runner import MPICH_MPIRunner
from libensemble.resources.env_resources import EnvResources
from libensemble.resources.mpi_resources import create_machinefile
from libensemble.resources.resources import GlobalResources, Resources, ResourcesException
from libensemble.resources.scheduler import ResourceScheduler
from libensemble.resources.worker_resources import ResourceManager, WorkerResources


def setup_standalone_run():
    os.environ.pop("LIBE_RESOURCES_TEST_NODE_LIST", None)
    os.environ.pop("LIBE_TEST_SLOTS_1", None)
    os.environ.pop("LIBE_TEST_GPUS_1", None)
    os.environ.pop("LIBE_TEST_GPUS_2", None)
    if os.path.isfile("node_list"):
        os.remove("node_list")


def teardown_standalone_run():
    os.environ.pop("LIBE_RESOURCES_TEST_NODE_LIST", None)
    os.environ.pop("LIBE_TEST_SLOTS_1", None)
    os.environ.pop("LIBE_TEST_GPUS_1", None)
    os.environ.pop("LIBE_TEST_GPUS_2", None)
    if os.path.isfile("node_list"):
        os.remove("node_list")


def setup_function(function):
    print(f"setup_function    function:{function.__name__}")
    os.environ.pop("LIBE_RESOURCES_TEST_NODE_LIST", None)
    os.environ.pop("LIBE_TEST_SLOTS_1", None)
    os.environ.pop("LIBE_TEST_GPUS_1", None)
    os.environ.pop("LIBE_TEST_GPUS_2", None)
    if os.path.isfile("node_list"):
        os.remove("node_list")


def teardown_function(function):
    print(f"teardown_function    function:{function.__name__}")
    os.environ.pop("LIBE_RESOURCES_TEST_NODE_LIST", None)
    os.environ.pop("LIBE_TEST_SLOTS_1", None)
    os.environ.pop("LIBE_TEST_GPUS_1", None)
    os.environ.pop("LIBE_TEST_GPUS_2", None)
    if os.path.isfile("node_list"):
        os.remove("node_list")


def sname(name):
    print("sname being set")
    return name.split(".", 1)[0]


# Tests ========================================================================================


# Tests GlobalResources.get_global_nodelist (This requires above tests to work)
def test_get_global_nodelist_frm_slurm():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "knl-[0020-0022,0137-0139,1234]"
    exp_out = ["knl-0020", "knl-0021", "knl-0022", "knl-0137", "knl-0138", "knl-0139", "knl-1234"]
    env_resources = EnvResources(
        nodelist_env_slurm="LIBE_RESOURCES_TEST_NODE_LIST",
        nodelist_env_cobalt="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_lsf="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_lsf_shortform="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
    )
    global_nodelist = GlobalResources.get_global_nodelist(rundir=os.getcwd(), env_resources=env_resources)
    assert global_nodelist == exp_out, "global_nodelist returned does not match expected. \nRet: {}\nExp: {}".format(
        global_nodelist, exp_out
    )


def test_get_global_nodelist_frm_slurm_suffix():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "knl-[0020-0022,0137-0139,1234].eps"
    exp_out = [
        "knl-0020.eps",
        "knl-0021.eps",
        "knl-0022.eps",
        "knl-0137.eps",
        "knl-0138.eps",
        "knl-0139.eps",
        "knl-1234.eps",
    ]
    env_resources = EnvResources(
        nodelist_env_slurm="LIBE_RESOURCES_TEST_NODE_LIST",
        nodelist_env_cobalt="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_lsf="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_lsf_shortform="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
    )
    global_nodelist = GlobalResources.get_global_nodelist(rundir=os.getcwd(), env_resources=env_resources)
    assert global_nodelist == exp_out, "global_nodelist returned does not match expected. \nRet: {}\nExp: {}".format(
        global_nodelist, exp_out
    )


def test_get_global_nodelist_frm_slurm_single():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "knl-0020"
    exp_out = ["knl-0020"]
    env_resources = EnvResources(
        nodelist_env_slurm="LIBE_RESOURCES_TEST_NODE_LIST",
        nodelist_env_cobalt="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_lsf="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_lsf_shortform="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
    )
    global_nodelist = GlobalResources.get_global_nodelist(rundir=os.getcwd(), env_resources=env_resources)
    assert global_nodelist == exp_out, "global_nodelist returned does not match expected. \nRet: {}\nExp: {}".format(
        global_nodelist, exp_out
    )


def test_get_global_nodelist_frm_slurm_straight_list():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "n0000.es1,n0002.es1"
    exp_out = ["n0000.es1", "n0002.es1"]
    env_resources = EnvResources(
        nodelist_env_slurm="LIBE_RESOURCES_TEST_NODE_LIST",
        nodelist_env_cobalt="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_lsf="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_lsf_shortform="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
    )
    global_nodelist = GlobalResources.get_global_nodelist(rundir=os.getcwd(), env_resources=env_resources)
    assert global_nodelist == exp_out, "global_nodelist returned does not match expected. \nRet: {}\nExp: {}".format(
        global_nodelist, exp_out
    )


# Default also sorts
def test_get_global_nodelist_frm_slurm_multigroup():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "knl-[0137-0139,1234],bdw[021,455]ext"
    exp_out = ["bdw021ext", "bdw455ext", "knl-0137", "knl-0138", "knl-0139", "knl-1234"]
    env_resources = EnvResources(
        nodelist_env_slurm="LIBE_RESOURCES_TEST_NODE_LIST",
        nodelist_env_cobalt="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_lsf="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_lsf_shortform="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
    )
    global_nodelist = GlobalResources.get_global_nodelist(rundir=os.getcwd(), env_resources=env_resources)
    assert global_nodelist == exp_out, "global_nodelist returned does not match expected. \nRet: {}\nExp: {}".format(
        global_nodelist, exp_out
    )


def test_get_global_nodelist_frm_slurm_complex():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "n0000.es1,knl-[0137-0139,1234],n0002.es1,bds[021,455]-ext"
    exp_out = ["bds021-ext", "bds455-ext", "knl-0137", "knl-0138", "knl-0139", "knl-1234", "n0000.es1", "n0002.es1"]
    env_resources = EnvResources(
        nodelist_env_slurm="LIBE_RESOURCES_TEST_NODE_LIST",
        nodelist_env_cobalt="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_lsf="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_lsf_shortform="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
    )
    global_nodelist = GlobalResources.get_global_nodelist(rundir=os.getcwd(), env_resources=env_resources)
    assert global_nodelist == exp_out, "global_nodelist returned does not match expected. \nRet: {}\nExp: {}".format(
        global_nodelist, exp_out
    )


def test_get_global_nodelist_frm_cobalt():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "20-22,137-139,1234"
    exp_out = ["20", "21", "22", "137", "138", "139", "1234"]
    env_resources = EnvResources(
        nodelist_env_slurm="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_cobalt="LIBE_RESOURCES_TEST_NODE_LIST",
        nodelist_env_lsf="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_lsf_shortform="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
    )
    global_nodelist = GlobalResources.get_global_nodelist(rundir=os.getcwd(), env_resources=env_resources)
    assert global_nodelist == exp_out, "global_nodelist returned does not match expected. \nRet: {}\nExp: {}".format(
        global_nodelist, exp_out
    )


def test_get_global_nodelist_frm_lsf():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "batch5" + " g06n02" * 42 + " h21n18" * 42
    exp_out = ["g06n02", "h21n18"]
    env_resources = EnvResources(
        nodelist_env_slurm="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_cobalt="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_lsf="LIBE_RESOURCES_TEST_NODE_LIST",
        nodelist_env_lsf_shortform="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
    )
    global_nodelist = GlobalResources.get_global_nodelist(rundir=os.getcwd(), env_resources=env_resources)
    assert global_nodelist == exp_out, "global_nodelist returned does not match expected. \nRet: {}\nExp: {}".format(
        global_nodelist, exp_out
    )


def test_get_global_nodelist_frm_lsf_shortform():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "batch5 1 g06n02 42 h21n18 42"
    exp_out = ["g06n02", "h21n18"]
    env_resources = EnvResources(
        nodelist_env_slurm="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_cobalt="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_lsf="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_lsf_shortform="LIBE_RESOURCES_TEST_NODE_LIST",
    )
    global_nodelist = GlobalResources.get_global_nodelist(rundir=os.getcwd(), env_resources=env_resources)
    assert global_nodelist == exp_out, "global_nodelist returned does not match expected. \nRet: {}\nExp: {}".format(
        global_nodelist, exp_out
    )


def test_get_global_nodelist_standalone():
    mynode = socket.gethostname()
    exp_node = mynode  # sname(mynode)
    env_resources = EnvResources(
        nodelist_env_slurm="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_cobalt="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_lsf="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_lsf_shortform="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
    )
    global_nodelist = GlobalResources.get_global_nodelist(rundir=os.getcwd(), env_resources=env_resources)
    assert global_nodelist == [exp_node], "global_nodelist returned does not match expected"


def test_get_global_nodelist_frm_wrklst_file(tmp_path):
    # node_list file should override env variables
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "20-22,137-139,1234"  # Should not be this
    exp_out = ["knl-0019", "knl-0021", "knl-0022", "knl-0137", "knl-0138", "knl-0139", "knl-2345"]  # Should be this

    open(tmp_path / "node_list", "w").close()
    try:
        _ = GlobalResources.get_global_nodelist(rundir=tmp_path)
    except ResourcesException as e:
        assert e.args[0] == "Error. global_nodelist is empty"
    else:
        assert 0

    with open(tmp_path / "node_list", "w") as f:
        for node in exp_out:
            f.write(node + "\n")

    # Do not specify env vars.
    global_nodelist1 = GlobalResources.get_global_nodelist(rundir=tmp_path)
    assert global_nodelist1 == exp_out, "global_nodelist returned does not match expected"

    # Specify env vars - should ignore
    env_resources = EnvResources(
        nodelist_env_slurm="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_cobalt="LIBE_RESOURCES_TEST_NODE_LIST",
        nodelist_env_lsf="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_lsf_shortform="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
    )
    global_nodelist2 = GlobalResources.get_global_nodelist(rundir=tmp_path, env_resources=env_resources)
    assert global_nodelist2 == exp_out, "global_nodelist returned does not match expected"
    os.remove(tmp_path / "node_list")


def test_remove_libE_nodes():
    mynode = socket.gethostname()
    exp_out = ["knl-0019", "knl-0021", "knl-0022", "knl-0137", "knl-0138", "knl-0139", "knl-2345"]

    # Add at beginning
    nodes_in = [mynode] + exp_out
    nodes_out = GlobalResources.remove_nodes(nodes_in, mynode)
    assert nodes_out == exp_out, "nodelist returned does not match expected"

    # Add twice in middle and at end
    nodes_in = []
    for i, node in enumerate(exp_out):
        nodes_in.append(node)
        if i == 1 or i == 4 or i == 6:
            nodes_in.append(mynode)
    nodes_out = GlobalResources.remove_nodes(nodes_in, mynode)
    assert nodes_out == exp_out, "nodelist returned does not match expected"


def _assert_worker_attr(wres, attr, exp):
    ret = getattr(wres, attr)
    assert ret == exp, f"{attr} returned does not match expected.  \nRet: {ret}\nExp: {exp}"


# These are all 1 worker per rset.
def _worker_asserts(wres, split_list, exp_slots, wrk, nworkers, nnodes, reps=1):
    # Create dictionary of attributes and expected values
    exp_dict = {
        "workerID": wrk + 1,
        "local_nodelist": split_list[wrk],
        "slots": exp_slots,
        "local_node_count": nnodes,
        "num_workers": nworkers,
        "split_list": split_list,
        "slot_count": 1,
        "local_rsets_list": [reps] * nworkers,
        "rsets_per_node": reps,
    }

    for attr, exp_val in exp_dict.items():
        _assert_worker_attr(wres, attr, exp_val)


# These are all >= 1 node per rset. And 1 worker per rset
# dedicated_mode makes no difference in this test
def test_get_local_resources_dedicated_mode():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "knl-[0020-0022,0036,0137-0139,1234]"
    resource_info = {"nodelist_env_slurm": "LIBE_RESOURCES_TEST_NODE_LIST"}
    libE_specs = {
        "resource_info": resource_info,
        "dedicated_mode": True,
    }
    gresources = GlobalResources(libE_specs)

    # 8 Workers ---------------------------------------------------------------
    nworkers = 8
    exp_out = [
        ["knl-0020"],
        ["knl-0021"],
        ["knl-0022"],
        ["knl-0036"],
        ["knl-0137"],
        ["knl-0138"],
        ["knl-0139"],
        ["knl-1234"],
    ]

    # Spoof current process as each worker and check nodelist and other worker resources.
    for wrk in range(nworkers):
        workerID = wrk + 1
        exp_slots = {exp_out[wrk][0]: [0]}
        wresources = WorkerResources(nworkers, gresources, workerID)
        wresources.set_rset_team([wrk])
        _worker_asserts(wresources, exp_out, exp_slots, wrk, nworkers, 1)
        del wresources

    # 4 Workers ---------------------------------------------------------------
    nworkers = 4
    exp_out = [
        ["knl-0020", "knl-0021"],
        ["knl-0022", "knl-0036"],
        ["knl-0137", "knl-0138"],
        ["knl-0139", "knl-1234"],
    ]
    for wrk in range(nworkers):
        workerID = wrk + 1
        exp_slots = {exp_out[wrk][0]: [0], exp_out[wrk][1]: [0]}
        wresources = WorkerResources(nworkers, gresources, workerID)
        wresources.set_rset_team([wrk])
        _worker_asserts(wresources, exp_out, exp_slots, wrk, nworkers, 2)
        del wresources

    # 1 Worker ----------------------------------------------------------------
    nworkers = 1
    exp_out = [["knl-0020", "knl-0021", "knl-0022", "knl-0036", "knl-0137", "knl-0138", "knl-0139", "knl-1234"]]

    # Just write out this one as one worker.
    exp_slots = {
        "knl-0020": [0],
        "knl-0021": [0],
        "knl-0022": [0],
        "knl-0036": [0],
        "knl-0137": [0],
        "knl-0138": [0],
        "knl-0139": [0],
        "knl-1234": [0],
    }

    for wrk in range(nworkers):
        workerID = wrk + 1
        # exp_slots - see above
        wresources = WorkerResources(nworkers, gresources, workerID)
        wresources.set_rset_team([wrk])
        _worker_asserts(wresources, exp_out, exp_slots, wrk, nworkers, 8)
        del wresources

    # 3 Workers (Test the best_split algorithm) -------------------------------
    nworkers = 3
    exp_out = [
        ["knl-0020", "knl-0021", "knl-0022"],
        ["knl-0036", "knl-0137", "knl-0138"],
        ["knl-0139", "knl-1234"],
    ]
    for wrk in range(nworkers):
        workerID = wrk + 1
        exp_slots = {exp_out[wrk][0]: [0], exp_out[wrk][1]: [0]}
        wresources = WorkerResources(nworkers, gresources, workerID)
        wresources.set_rset_team([wrk])

        # Trying to avoid generic algorithm
        if len(wresources.local_nodelist) == 3:
            exp_slots = {exp_out[wrk][0]: [0], exp_out[wrk][1]: [0], exp_out[wrk][2]: [0]}
            nnodes = 3
        else:
            exp_slots = {exp_out[wrk][0]: [0], exp_out[wrk][1]: [0]}
            nnodes = 2
        _worker_asserts(wresources, exp_out, exp_slots, wrk, nworkers, nnodes)

    # 16 Workers --------------------------------------------------------------
    # Multiple workers per node
    nworkers = 16

    exp_out = [
        ["knl-0020"],
        ["knl-0020"],
        ["knl-0021"],
        ["knl-0021"],
        ["knl-0022"],
        ["knl-0022"],
        ["knl-0036"],
        ["knl-0036"],
        ["knl-0137"],
        ["knl-0137"],
        ["knl-0138"],
        ["knl-0138"],
        ["knl-0139"],
        ["knl-0139"],
        ["knl-1234"],
        ["knl-1234"],
    ]

    for wrk in range(nworkers):
        workerID = wrk + 1

        # If even worker have slot 1
        if (workerID % 2) == 0:
            myslot = 1
        else:
            myslot = 0
        exp_slots = {exp_out[wrk][0]: [myslot]}
        wresources = WorkerResources(nworkers, gresources, workerID)
        wresources.set_rset_team([wrk])
        _worker_asserts(wresources, exp_out, exp_slots, wrk, nworkers, 1, 2)

    del os.environ["LIBE_RESOURCES_TEST_NODE_LIST"]


# The main tests are same as above - note for when fixtures set up
def test_get_local_resources_dedicated_mode_remove_libE_proc(tmp_path):
    mynode = socket.gethostname()
    nodelist_in = ["knl-0020", "knl-0021", "knl-0022", "knl-0036", "knl-0137", "knl-0138", "knl-0139", "knl-1234"]
    with open(tmp_path / "node_list", "w") as f:
        for i, node in enumerate(nodelist_in):
            f.write(node + "\n")
            if i == 3:
                f.write(mynode + "\n")

    libE_specs = {"dedicated_mode": True}

    gresources = GlobalResources(libE_specs, top_level_dir=tmp_path)
    gresources.add_comm_info(libE_nodes=[mynode])

    # 8 Workers ---------------------------------------------------------------
    nworkers = 8
    exp_out = [
        ["knl-0020"],
        ["knl-0021"],
        ["knl-0022"],
        ["knl-0036"],
        ["knl-0137"],
        ["knl-0138"],
        ["knl-0139"],
        ["knl-1234"],
    ]

    # Spoof current process as each worker and check nodelist and other worker resources.
    for wrk in range(nworkers):
        workerID = wrk + 1
        exp_slots = {exp_out[wrk][0]: [0]}
        wresources = WorkerResources(nworkers, gresources, workerID)
        wresources.set_rset_team([wrk])
        _worker_asserts(wresources, exp_out, exp_slots, wrk, nworkers, 1)
        del wresources

    # 4 Workers ---------------------------------------------------------------
    nworkers = 4
    exp_out = [
        ["knl-0020", "knl-0021"],
        ["knl-0022", "knl-0036"],
        ["knl-0137", "knl-0138"],
        ["knl-0139", "knl-1234"],
    ]
    for wrk in range(nworkers):
        workerID = wrk + 1
        exp_slots = {exp_out[wrk][0]: [0], exp_out[wrk][1]: [0]}
        wresources = WorkerResources(nworkers, gresources, workerID)
        wresources.set_rset_team([wrk])
        _worker_asserts(wresources, exp_out, exp_slots, wrk, nworkers, 2)
        del wresources

    # 1 Worker ----------------------------------------------------------------
    nworkers = 1
    exp_out = [["knl-0020", "knl-0021", "knl-0022", "knl-0036", "knl-0137", "knl-0138", "knl-0139", "knl-1234"]]

    # Just write out this one as one worker.
    exp_slots = {
        "knl-0020": [0],
        "knl-0021": [0],
        "knl-0022": [0],
        "knl-0036": [0],
        "knl-0137": [0],
        "knl-0138": [0],
        "knl-0139": [0],
        "knl-1234": [0],
    }

    for wrk in range(nworkers):
        workerID = wrk + 1
        # exp_slots - see above
        wresources = WorkerResources(nworkers, gresources, workerID)
        wresources.set_rset_team([wrk])
        _worker_asserts(wresources, exp_out, exp_slots, wrk, nworkers, 8)
        del wresources

    # 3 Workers (Test the best_split algorithm) -------------------------------
    nworkers = 3
    exp_out = [
        ["knl-0020", "knl-0021", "knl-0022"],
        ["knl-0036", "knl-0137", "knl-0138"],
        ["knl-0139", "knl-1234"],
    ]
    for wrk in range(nworkers):
        workerID = wrk + 1
        exp_slots = {exp_out[wrk][0]: [0], exp_out[wrk][1]: [0]}
        wresources = WorkerResources(nworkers, gresources, workerID)
        wresources.set_rset_team([wrk])

        # Trying to avoid generic algorithm
        if len(wresources.local_nodelist) == 3:
            exp_slots = {exp_out[wrk][0]: [0], exp_out[wrk][1]: [0], exp_out[wrk][2]: [0]}
            nnodes = 3
        else:
            exp_slots = {exp_out[wrk][0]: [0], exp_out[wrk][1]: [0]}
            nnodes = 2
        _worker_asserts(wresources, exp_out, exp_slots, wrk, nworkers, nnodes)
        del wresources

    # 16 Workers --------------------------------------------------------------
    # Multiple workers per node
    nworkers = 16

    exp_out = [
        ["knl-0020"],
        ["knl-0020"],
        ["knl-0021"],
        ["knl-0021"],
        ["knl-0022"],
        ["knl-0022"],
        ["knl-0036"],
        ["knl-0036"],
        ["knl-0137"],
        ["knl-0137"],
        ["knl-0138"],
        ["knl-0138"],
        ["knl-0139"],
        ["knl-0139"],
        ["knl-1234"],
        ["knl-1234"],
    ]

    for wrk in range(nworkers):
        workerID = wrk + 1

        # If even worker have slot 1
        if (workerID % 2) == 0:
            myslot = 1
        else:
            myslot = 0
        exp_slots = {exp_out[wrk][0]: [myslot]}
        wresources = WorkerResources(nworkers, gresources, workerID)
        wresources.set_rset_team([wrk])
        _worker_asserts(wresources, exp_out, exp_slots, wrk, nworkers, 1, 2)
        del wresources

    os.remove(tmp_path / "node_list")


def test_get_local_nodelist_distrib_mode_host_not_in_list():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "knl-[0020-0022,0036,0137-0139,1234]"
    resource_info = {"nodelist_env_slurm": "LIBE_RESOURCES_TEST_NODE_LIST"}
    libE_specs = {"resource_info": resource_info, "dedicated_mode": False}

    gresources = GlobalResources(libE_specs)
    nworkers = 4
    exp_out = ["knl-0022", "knl-0036"]

    # Test running distributed mode without current host in list.
    workerID = 2
    wresources = WorkerResources(nworkers, gresources, workerID)
    wresources.set_rset_team([workerID - 1])
    assert wresources.local_nodelist == exp_out, "local_nodelist returned does not match expected"
    del wresources


def test_get_local_nodelist_distrib_mode(tmp_path):
    mynode = socket.gethostname()
    # nodelist_in = ['knl-0020', 'knl-0021', 'knl-0022', 'knl-0036', 'knl-0137', 'knl-0138', 'knl-0139', 'knl-1234']
    nodelist_in = ["knl-0020", "knl-0021", "knl-0022", "knl-0036", "knl-0137", "knl-0138", "knl-0139"]
    with open(tmp_path / "node_list", "w") as f:
        for i, node in enumerate(nodelist_in):
            f.write(node + "\n")
            if i == 3:
                f.write(mynode + "\n")

    libE_specs = {"dedicated_mode": False}
    gresources = GlobalResources(libE_specs, top_level_dir=tmp_path)
    gresources.add_comm_info(libE_nodes=[mynode])

    # Spoof current process as each worker and check nodelist.
    nworkers = 8
    workerID = 5
    exp_node = mynode  # sname(mynode)
    exp_out = [exp_node]
    exp_slots = {exp_node: [0]}
    wresources = WorkerResources(nworkers, gresources, workerID)
    wresources.set_rset_team([workerID - 1])
    assert wresources.local_nodelist == exp_out, "local_nodelist returned does not match expected"

    # Do the rest
    exp_split = [
        ["knl-0020"],
        ["knl-0021"],
        ["knl-0022"],
        ["knl-0036"],
        [exp_node],
        ["knl-0137"],
        ["knl-0138"],
        ["knl-0139"],
    ]
    _worker_asserts(wresources, exp_split, exp_slots, workerID - 1, nworkers, 1)
    del wresources

    nworkers = 1
    workerID = 1
    wresources = WorkerResources(nworkers, gresources, workerID)
    wresources.set_rset_team([workerID - 1])
    exp_out = ["knl-0020", "knl-0021", "knl-0022", "knl-0036", exp_node, "knl-0137", "knl-0138", "knl-0139"]
    assert wresources.local_nodelist == exp_out, "local_nodelist returned does not match expected"
    del wresources

    nworkers = 4
    workerID = 3
    wresources = WorkerResources(nworkers, gresources, workerID)
    wresources.set_rset_team([workerID - 1])
    exp_out = [exp_node, "knl-0137"]
    assert wresources.local_nodelist == exp_out, "local_nodelist returned does not match expected"
    del wresources

    # Sub-node workers
    nworkers = 16
    workerID = 9
    wresources = WorkerResources(nworkers, gresources, workerID)
    wresources.set_rset_team([workerID - 1])
    exp_out = [exp_node]
    assert wresources.local_nodelist == exp_out, "local_nodelist returned does not match expected"
    del wresources

    workerID = 10
    wresources = WorkerResources(nworkers, gresources, workerID)
    wresources.set_rset_team([workerID - 1])
    exp_out = [exp_node]
    assert wresources.local_nodelist == exp_out, "local_nodelist returned does not match expected"
    del wresources

    os.remove(tmp_path / "node_list")


def test_get_local_nodelist_distrib_mode_uneven_split(tmp_path):
    mynode = socket.gethostname()
    exp_node = mynode  # sname(mynode)
    nodelist_in = ["knl-0020", "knl-0021", "knl-0022", "knl-0036", "knl-0137", "knl-0138", "knl-0139", "knl-1234"]
    with open(tmp_path / "node_list", "w") as f:
        for i, node in enumerate(nodelist_in):
            f.write(node + "\n")
            if i == 4:
                f.write(mynode + "\n")

    libE_specs = {"dedicated_mode": False}
    gresources = GlobalResources(libE_specs, top_level_dir=tmp_path)
    gresources.add_comm_info(libE_nodes=[mynode])
    nworkers = 2

    # May not be at head of list - should perhaps be warning or enforced
    exp_out_w1 = ["knl-0020", "knl-0021", "knl-0022", "knl-0036", "knl-0137"]
    exp_out_w2 = [exp_node, "knl-0138", "knl-0139", "knl-1234"]
    exp_split = [exp_out_w1, exp_out_w2]

    workerID = 2
    wresources = WorkerResources(nworkers, gresources, workerID)
    wresources.set_rset_team([workerID - 1])
    exp_slots = {exp_node: [0], "knl-0138": [0], "knl-0139": [0], "knl-1234": [0]}
    _worker_asserts(wresources, exp_split, exp_slots, workerID - 1, nworkers, 4)
    del wresources

    workerID = 1
    wresources = WorkerResources(nworkers, gresources, workerID)
    wresources.set_rset_team([workerID - 1])
    exp_slots = {"knl-0020": [0], "knl-0021": [0], "knl-0022": [0], "knl-0036": [0], "knl-0137": [0]}
    _worker_asserts(wresources, exp_split, exp_slots, workerID - 1, nworkers, 5)
    del wresources

    os.remove(tmp_path / "node_list")


def test_get_group_list():
    # 8 resource sets on different nodes --------------------------------------
    split_list = [
        ["knl-0020"],
        ["knl-0021"],
        ["knl-0022"],
        ["knl-0036"],
        ["knl-0137"],
        ["knl-0138"],
        ["knl-0139"],
        ["knl-1234"],
    ]
    group_list, slot_list, _ = ResourceManager.get_group_list(split_list)
    assert group_list == [1, 2, 3, 4, 5, 6, 7, 8]
    assert slot_list == [0, 0, 0, 0, 0, 0, 0, 0]

    # 4 resource sets with 2 nodes each ---------------------------------------
    split_list = [
        ["knl-0020", "knl-0021"],
        ["knl-0022", "knl-0036"],
        ["knl-0137", "knl-0138"],
        ["knl-0139", "knl-1234"],
    ]
    group_list, slot_list, _ = ResourceManager.get_group_list(split_list)
    assert group_list == [1, 2, 3, 4]
    assert slot_list == [0, 0, 0, 0]

    # 8 resource sets - 2 per node --------------------------------------------
    split_list = [
        ["knl-0020"],
        ["knl-0020"],
        ["knl-0022"],
        ["knl-0022"],
        ["knl-0137"],
        ["knl-0137"],
        ["knl-0139"],
        ["knl-0139"],
    ]
    group_list, slot_list, _ = ResourceManager.get_group_list(split_list)
    assert group_list == [1, 1, 2, 2, 3, 3, 4, 4]
    assert slot_list == [0, 1, 0, 1, 0, 1, 0, 1]

    # 8 resource sets - uneven ------------------------------------------------
    split_list = [
        ["knl-0020"],
        ["knl-0020"],
        ["knl-0020"],
        ["knl-0020"],
        ["knl-0137"],
        ["knl-0137"],
        ["knl-0137"],
        ["knl-0139"],
    ]
    group_list, slot_list, _ = ResourceManager.get_group_list(split_list)
    assert group_list == [1, 1, 1, 1, 2, 2, 2, 3]
    assert slot_list == [0, 1, 2, 3, 0, 1, 2, 0]


def _check_mfile(machinefile, exp_list):
    with open("machinefile", "r") as f:
        i = 0
        for line in f:
            index = i // 4
            assert line == exp_list[index]
            i += 1


def test_machinefile_from_resources():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "knl-[0020-0022,0036,0137-0139,1234]"
    resource_info = {"nodelist_env_slurm": "LIBE_RESOURCES_TEST_NODE_LIST"}
    libE_specs = {
        "resource_info": resource_info,
        "num_resource_sets": 8,
    }

    exp_list = ["knl-0020\n", "knl-0021\n", "knl-0022\n", "knl-0036\n"]

    resources = Resources(libE_specs)
    resources.set_worker_resources(4, 1)
    resources.worker_resources.set_rset_team([0, 1, 2, 3])

    built_mfile = create_machinefile(resources, num_nodes=4, procs_per_node=4)
    assert built_mfile, "machinefile doesn't exist or is empty"

    _check_mfile("machinefile", exp_list)

    # Test replacing older machinefile
    create_machinefile(resources, machinefile="machinefile", num_nodes=4, procs_per_node=4)
    _check_mfile("machinefile", exp_list)
    os.remove("machinefile")


def _setup_wresources_gpus(ngpus, gpu_sharing_factor=1, num_resource_sets=None, gpus_per_group=None):
    """Set up worker resources with a given number of GPUs"""
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "knl-[0020,0036]"

    resource_info = {"nodelist_env_slurm": "LIBE_RESOURCES_TEST_NODE_LIST", "gpus_on_node": ngpus}
    libE_specs = {
        "resource_info": resource_info,
        "dedicated_mode": False,
        "gpu_sharing_factor": gpu_sharing_factor,
        "gpus_per_group": gpus_per_group,
    }
    if num_resource_sets is not None:
        libE_specs["num_resource_sets"] = num_resource_sets
    gresources = GlobalResources(libE_specs)
    nworkers = 8
    workerID = 2  # Does not matter as using set_rset_team
    wresources = WorkerResources(nworkers, gresources, workerID)
    return wresources


def _setup_wresources_gpus_frm_platform_info(ngpus, ntiles):
    """Set up worker resources with a given number of GPUs"""
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "knl-[0020,0036]"
    resource_info = {"nodelist_env_slurm": "LIBE_RESOURCES_TEST_NODE_LIST"}

    platform_info = {
        "gpu_setting_type": "env",
        "gpu_setting_name": "LIBE_TEST_GPUS_1",
        "scheduler_match_slots": True,
        "gpus_per_node": ngpus,
        "tiles_per_gpu": ntiles,
    }

    libE_specs = {"resource_info": resource_info, "use_tiles_as_gpus": True}
    gresources = GlobalResources(libE_specs, platform_info)
    nworkers = 24
    workerID = 2  # Does not matter as using set_rset_team
    wresources = WorkerResources(nworkers, gresources, workerID)
    return wresources


def test_wresources_set_gpus():
    wresources = _setup_wresources_gpus(4)

    rset_teams = [[0, 1, 4, 5], [2, 3, 6, 7], [1, 3, 5, 7]]
    exp_out = ["0,1", "2,3", "1,3"]

    # Test routines for querying slots and gpus.
    for i in range(3):
        wresources.set_rset_team(rset_teams[i])

        slots = wresources.get_slots_as_string()
        assert slots == exp_out[i], f"Slots {slots} does not match expected {exp_out[i]}"

        wresources.set_env_to_slots("LIBE_TEST_SLOTS_1")
        eslots = os.environ["LIBE_TEST_SLOTS_1"]
        assert eslots == exp_out[i], f"Env slots {eslots} does not match expected {exp_out[i]}"

        wresources.set_env_to_gpus("LIBE_TEST_GPUS_1")
        egpus = os.environ["LIBE_TEST_GPUS_1"]
        assert egpus == exp_out[i], f"Env GPUs {egpus} does not match expected {exp_out[i]}"

        os.environ.pop("LIBE_TEST_SLOTS_1", None)
        os.environ.pop("LIBE_TEST_GPUS_1", None)


def _check_set_env_to_gpus(wresources, env, exp):
    """Check set_env_to_gpus for given wresources/platform"""
    os.environ.pop(env, None)
    wresources.set_env_to_gpus()
    egpus = os.environ[env]
    assert egpus == exp, f"Env GPUs {egpus} does not match expected {exp}"
    os.environ.pop(env, None)


def test_wresources_set_gpus_by_platform():
    wresources = _setup_wresources_gpus(4)

    rset_teams = [[0, 1, 4, 5], [2, 3, 6, 7], [1, 3, 5, 7]]
    exp_out = ["0,1", "2,3", "1,3"]

    # Test routines for querying slots and gpus.
    for i in range(3):
        wresources.set_rset_team(rset_teams[i])

        wresources.platform_info = {
            "gpu_setting_type": "env",
            "gpu_setting_name": "LIBE_TEST_GPUS_1",
            "scheduler_match_slots": False,
        }

        _check_set_env_to_gpus(wresources, "LIBE_TEST_GPUS_1", exp_out[i])
        os.environ.pop("LIBE_TEST_GPUS_1", None)

        # "gpu_setting_type" should override "gpu_env_fallback"
        wresources.platform_info["gpu_env_fallback"] = "LIBE_TEST_GPUS_2"  # should ignore
        _check_set_env_to_gpus(wresources, "LIBE_TEST_GPUS_1", exp_out[i])
        assert "LIBE_TEST_GPUS_2" not in os.environ, "LIBE_TEST_GPUS_2 should not be set"

        # Change "gpu_setting_type" so not "env" and it should use "gpu_env_fallback"
        wresources.platform_info["gpu_setting_type"] = "runner_default"

        _check_set_env_to_gpus(wresources, "LIBE_TEST_GPUS_2", exp_out[i])
        assert "LIBE_TEST_GPUS_1" not in os.environ, "LIBE_TEST_GPUS_1 should not be set"

        # Try default - via empty string or not set
        wresources.platform_info["gpu_env_fallback"] = ""
        _check_set_env_to_gpus(wresources, "CUDA_VISIBLE_DEVICES", exp_out[i])

        del wresources.platform_info["gpu_env_fallback"]
        _check_set_env_to_gpus(wresources, "CUDA_VISIBLE_DEVICES", exp_out[i])

        # If no platform info.
        wresources.platform_info = None
        _check_set_env_to_gpus(wresources, "CUDA_VISIBLE_DEVICES", exp_out[i])
    del wresources


def test_wresources_set_gpus_by_platform_gpu_tiles():
    """
    Resource set IDs
    Node 1:   0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11
    Node 2:  12,13,14,15,16,17,18,19,20,21,22,23

    # Convert resource sets to slot IDs (0,1,2,3) and then to tiles
    [0, 1, 2, 3, 12, 13, 14, 15]
    = "0.0, 0.1, 1.0, 1.1"
    """
    wresources = _setup_wresources_gpus_frm_platform_info(6, 2)

    rset_teams = [[0, 1, 2, 3, 12, 13, 14, 15], [3, 6, 8, 15, 18, 20]]
    exp_out = ["0.0,0.1,1.0,1.1", "1.1,3.0,4.0"]

    # Test routines for querying slots and gpus.
    for i in range(2):
        wresources.set_rset_team(rset_teams[i])

        _check_set_env_to_gpus(wresources, "LIBE_TEST_GPUS_1", exp_out[i])
        os.environ.pop("LIBE_TEST_GPUS_1", None)
    del wresources


def test_wresources_set_gpus_with_sharing():
    wresources = _setup_wresources_gpus(4, gpu_sharing_factor=2, num_resource_sets=16)

    assert wresources.total_num_gpu_rsets == 16
    assert list(wresources.all_rsets["gpu_id"]) == [0, 0, 1, 1, 2, 2, 3, 3] * 2
    assert list(wresources.all_rsets["gpu_share"]) == [0, 1, 0, 1, 0, 1, 0, 1] * 2

    wresources.set_rset_team([0, 1, 2, 3])
    wresources.set_env_to_gpus("LIBE_TEST_GPUS_1")
    assert os.environ["LIBE_TEST_GPUS_1"] == "0,1"
    os.environ.pop("LIBE_TEST_GPUS_1")


def test_gpu_sharing_executor_env_mapping():
    wresources = _setup_wresources_gpus(4, gpu_sharing_factor=2, num_resource_sets=16)
    wresources.set_rset_team([1])

    class Task:
        def _add_to_env(self, name, value):
            self.env = {name: value}

    task = Task()
    MPICH_MPIRunner()._set_gpu_env_var(wresources, task, 1, "CUDA_VISIBLE_DEVICES")
    assert task.env == {"CUDA_VISIBLE_DEVICES": "0"}


def test_gpu_sharing_mapping_with_gpu_groups():
    wresources = _setup_wresources_gpus(4, gpu_sharing_factor=2, num_resource_sets=16, gpus_per_group=2)

    assert list(wresources.all_rsets["slot"]) == [0, 1, 2, 3] * 4
    assert list(wresources.all_rsets["gpu_id"]) == [0, 0, 1, 1, 2, 2, 3, 3] * 2
    assert list(wresources.all_rsets["gpu_share"]) == [0, 1, 0, 1, 0, 1, 0, 1] * 2


def test_wresources_set_gpus_x2():
    wresources = _setup_wresources_gpus(8)

    rset_teams = [[0, 1, 4, 5], [2, 3, 6, 7], [1, 3, 5, 7]]
    exp_out = ["0,1,2,3", "4,5,6,7", "2,3,6,7"]

    # Test routines for querying slots and gpus.
    for i in range(3):
        wresources.set_rset_team(rset_teams[i])

        slots = wresources.get_slots_as_string(multiplier=2)
        assert slots == exp_out[i], f"Slots {slots} does not match expected {exp_out[i]}"

        wresources.set_env_to_slots("LIBE_TEST_SLOTS_1", multiplier=2)
        eslots = os.environ["LIBE_TEST_SLOTS_1"]
        assert eslots == exp_out[i], f"Env slots {eslots} does not match expected {exp_out[i]}"

        wresources.set_env_to_gpus("LIBE_TEST_GPUS_1")
        egpus = os.environ["LIBE_TEST_GPUS_1"]
        assert egpus == exp_out[i], f"Env GPUs {egpus} does not match expected {exp_out[i]}"

        del os.environ["LIBE_TEST_SLOTS_1"]
        del os.environ["LIBE_TEST_GPUS_1"]
    del wresources


def test_wresources_set_limit_gpus():
    wresources = _setup_wresources_gpus(12)

    rset_teams = [[0, 1, 4, 5], [2, 3, 6, 7], [1, 3, 5, 7]]
    # exp_out_0 = ["0,1,2,3,4,5", "6,7,8,9,10,11", "3,4,5,9,10,11"]
    exp_out_1 = ["0,1,2,3", "6,7,8,9", "3,4,5,9"]
    exp_out_2 = ["0", "6", "3"]

    # Test routines for querying slots and gpus.
    for i in range(3):
        wresources.set_rset_team(rset_teams[i])

        wresources.gen_ngpus = 4
        wresources.set_env_to_gpus("LIBE_TEST_GPUS_1")
        egpus = os.environ["LIBE_TEST_GPUS_1"]
        assert egpus == exp_out_1[i], f"Env GPUs {egpus} does not match expected {exp_out_1[i]}"

        wresources.gen_ngpus = 1
        wresources.set_env_to_gpus("LIBE_TEST_GPUS_1")
        egpus = os.environ["LIBE_TEST_GPUS_1"]
        assert egpus == exp_out_2[i], f"Env GPUs {egpus} does not match expected {exp_out_2[i]}"

        del os.environ["LIBE_TEST_GPUS_1"]
    del wresources


def test_wresources_set_no_gpus():
    wresources = _setup_wresources_gpus(0)
    rset_team = [0, 1, 4, 5]
    wresources.set_rset_team(rset_team)
    wresources.set_env_to_gpus("LIBE_TEST_GPUS_1")
    assert "LIBE_TEST_GPUS_1" not in os.environ


def _setup_resources_sharing_for_alloc(gpu_sharing_factor=2, num_resource_sets=16):
    """Create mocked Resources for AllocSupport fractional tests.

    Mocks a 2-node SLURM nodelist with 4 GPUs/node and sf=2, yielding
    8 GPU shares per node (16 total). Returns the ``Resources`` object that
    AllocSupport will use to convert fractional ``num_gpus`` to shares via
    ``ceil(num_gpus*sf/gpus_per_rset)``.
    """
    # Mock a 2-node SLURM nodelist; no real hardware needed
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "knl-[0020,0036]"
    resource_info = {"nodelist_env_slurm": "LIBE_RESOURCES_TEST_NODE_LIST", "gpus_on_node": 4}
    libE_specs = {
        "resource_info": resource_info,
        "dedicated_mode": False,  # keep libEnsemble ranks out of the nodelist
        "gpu_sharing_factor": gpu_sharing_factor,
        "num_resource_sets": num_resource_sets,
    }
    # Instantiate Resources and its ResourceManager (holds RSetResources)
    res = Resources(libE_specs=libE_specs, top_level_dir=os.getcwd())
    res.set_resource_manager(num_workers=8)
    return res


def test_alloc_support_fractional():
    """Unit coverage for ``AllocSupport`` fractional ``num_gpus`` handling.

    When ``gpu_sharing_factor>1`` the manager converts a fractional
    ``num_gpus`` (physical GPUs) to logical shares via
    ``ceil(num_gpus * sf / gpus_per_rset)``. This test drives:

    * ``_req_resources_sim`` with History rows ``0.5→1 share``, ``1.5→3``,
      ``2.0→4`` and asserts ``use_gpus`` and ``libE_info[\"num_gpus\"]``.
    * ``_req_resources_gen`` with ``persis_info[\"gen_num_gpus\"]`` ``0.5``
      and ``1.5`` (the ``def_gen_num_gpus`` path) likewise mapping to 1 and
      3 shares.

    Uses a lightweight ``AllocSupport`` instantiation with a mocked
    ``Resources`` object to avoid a full ensemble run.
    """
    # Use a mocked 16-rset Resources with sf=2 (gpus_per_rset=1)
    res = _setup_resources_sharing_for_alloc()
    from libensemble.tools.alloc_support import AllocSupport

    # Minimal worker array needed by AllocSupport
    W = np.zeros(3, dtype=[("persis_state", int), ("active", int)])
    libE_info = {}
    user_params: list = []
    # History with fractional num_gpus: 0.5 and 1.5 physical GPUs
    H = np.zeros(2, dtype=[("num_gpus", float), ("num_procs", int)])
    H["num_gpus"] = [0.5, 1.5]
    H["num_procs"] = [1, 1]
    # Pass the mocked Resources directly (avoid global Resources.resources=None issue)
    support = AllocSupport(W, manage_resources=True, libE_info=libE_info, user_resources=res)
    # Override scheduler to enforce pack policy for deterministic shares
    support.sched = ResourceScheduler(user_resources=res.resource_manager, sched_opts={"gpu_packing": "pack"})
    # 0.5 GPUs * sf2 / gpus_per_rset1 => ceil(1) = 1 share
    nsets, use_gpus = support._req_resources_sim(libE_info, user_params, H, [0])
    assert nsets == 1 and use_gpus is True and libE_info["num_gpus"] == 0.5
    user_params = []
    libE_info = {}
    # 1.5 GPUs => ceil(3) = 3 shares (1 whole GPU +1 share)
    nsets2, _ = support._req_resources_sim(libE_info, user_params, H, [1])
    assert nsets2 == 3
    user_params = []
    libE_info = {}
    # 2.0 GPUs => ceil(4) = 4 shares (2 whole GPUs)
    H2 = np.zeros(1, dtype=[("num_gpus", float)])
    H2["num_gpus"] = [2.0]
    nsets3, _ = support._req_resources_sim(libE_info, user_params, H2, [0])
    assert nsets3 == 4
    user_params = []
    libE_info = {}
    # Generator path via persis_info gen_num_gpus (0.5 => 1 share)
    support2 = AllocSupport(
        W, manage_resources=True, persis_info={"gen_num_gpus": 0.5}, libE_info={}, user_resources=support.resources
    )
    support2.resources = support.resources
    support2.sched = support.sched
    nsets4, _ = support2._req_resources_gen(libE_info, user_params)
    assert nsets4 == 1
    user_params = []
    libE_info = {}
    # Generator with 1.5 GPUs => 3 shares
    support2.persis_info = {"gen_num_gpus": 1.5}
    nsets5, _ = support2._req_resources_gen(libE_info, user_params)
    assert nsets5 == 3


if __name__ == "__main__":
    setup_standalone_run()

    test_get_global_nodelist_frm_slurm()
    test_get_global_nodelist_frm_slurm_suffix()
    test_get_global_nodelist_frm_slurm_single()
    test_get_global_nodelist_frm_slurm_straight_list()
    test_get_global_nodelist_frm_slurm_multigroup()
    test_get_global_nodelist_frm_slurm_complex()

    test_get_global_nodelist_frm_cobalt()
    test_get_global_nodelist_frm_lsf()
    test_get_global_nodelist_frm_lsf_shortform()
    test_get_global_nodelist_standalone()

    test_get_global_nodelist_frm_wrklst_file()
    test_remove_libE_nodes()

    test_get_local_resources_dedicated_mode()
    test_get_local_resources_dedicated_mode_remove_libE_proc()
    test_get_local_nodelist_distrib_mode_host_not_in_list()
    test_get_local_nodelist_distrib_mode()
    test_get_local_nodelist_distrib_mode_uneven_split()

    test_get_group_list()
    test_machinefile_from_resources()
    test_wresources_set_gpus()
    test_wresources_set_gpus_by_platform()
    test_wresources_set_gpus_by_platform_gpu_tiles()
    test_wresources_set_gpus_x2()
    test_wresources_set_limit_gpus()
    test_wresources_set_no_gpus()

    test_alloc_support_fractional()

    teardown_standalone_run()
