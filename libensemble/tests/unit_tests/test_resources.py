import os
import socket
from libensemble.resources.env_resources import EnvResources
from libensemble.resources.resources import Resources, GlobalResources, ResourcesException
from libensemble.resources.worker_resources import ResourceManager, WorkerResources
from libensemble.resources.mpi_resources import create_machinefile


def setup_standalone_run():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = ""
    if os.path.isfile("node_list"):
        os.remove("node_list")


def teardown_standalone_run():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = ""
    if os.path.isfile("node_list"):
        os.remove("node_list")


def setup_function(function):
    print(f"setup_function    function:{function.__name__}")
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = ""
    # if os.environ['LIBE_RESOURCES_TEST_NODE_LIST']:
    #     del os.environ['LIBE_RESOURCES_TEST_NODE_LIST']
    # if os.environ['THIS_ENV_VARIABLE_IS_DEF_NOT_SET']:
    #     del os.environ['THIS_ENV_VARIABLE_IS_DEF_NOT_SET']
    if os.path.isfile("node_list"):
        os.remove("node_list")


def teardown_function(function):
    print(f"teardown_function    function:{function.__name__}")
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = ""
    # if os.environ['LIBE_RESOURCES_TEST_NODE_LIST']:
    #     del os.environ['LIBE_RESOURCES_TEST_NODE_LIST']
    # if os.environ['THIS_ENV_VARIABLE_IS_DEF_NOT_SET']:
    #     del os.environ['THIS_ENV_VARIABLE_IS_DEF_NOT_SET']
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


def test_get_global_nodelist_frm_wrklst_file():
    # node_list file should override env variables
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "20-22,137-139,1234"  # Should not be this
    exp_out = ["knl-0019", "knl-0021", "knl-0022", "knl-0137", "knl-0138", "knl-0139", "knl-2345"]  # Should be this

    open("node_list", "w").close()
    try:
        _ = GlobalResources.get_global_nodelist(rundir=os.getcwd())
    except ResourcesException as e:
        assert e.args[0] == "Error. global_nodelist is empty"
    else:
        assert 0

    with open("node_list", "w") as f:
        for node in exp_out:
            f.write(node + "\n")

    # Do not specify env vars.
    global_nodelist1 = GlobalResources.get_global_nodelist(rundir=os.getcwd())
    assert global_nodelist1 == exp_out, "global_nodelist returned does not match expected"

    # Specify env vars - should ignore
    env_resources = EnvResources(
        nodelist_env_slurm="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_cobalt="LIBE_RESOURCES_TEST_NODE_LIST",
        nodelist_env_lsf="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
        nodelist_env_lsf_shortform="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
    )
    global_nodelist2 = GlobalResources.get_global_nodelist(rundir=os.getcwd(), env_resources=env_resources)
    assert global_nodelist2 == exp_out, "global_nodelist returned does not match expected"
    os.remove("node_list")


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
def test_get_local_resources_dedicated_mode_remove_libE_proc():
    mynode = socket.gethostname()
    nodelist_in = ["knl-0020", "knl-0021", "knl-0022", "knl-0036", "knl-0137", "knl-0138", "knl-0139", "knl-1234"]
    with open("node_list", "w") as f:
        for i, node in enumerate(nodelist_in):
            f.write(node + "\n")
            if i == 3:
                f.write(mynode + "\n")

    libE_specs = {"dedicated_mode": True}

    gresources = GlobalResources(libE_specs)
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

    os.remove("node_list")


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


def test_get_local_nodelist_distrib_mode():
    mynode = socket.gethostname()
    # nodelist_in = ['knl-0020', 'knl-0021', 'knl-0022', 'knl-0036', 'knl-0137', 'knl-0138', 'knl-0139', 'knl-1234']
    nodelist_in = ["knl-0020", "knl-0021", "knl-0022", "knl-0036", "knl-0137", "knl-0138", "knl-0139"]
    with open("node_list", "w") as f:
        for i, node in enumerate(nodelist_in):
            f.write(node + "\n")
            if i == 3:
                f.write(mynode + "\n")

    libE_specs = {"dedicated_mode": False}
    gresources = GlobalResources(libE_specs)
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

    os.remove("node_list")


def test_get_local_nodelist_distrib_mode_uneven_split():
    mynode = socket.gethostname()
    exp_node = mynode  # sname(mynode)
    nodelist_in = ["knl-0020", "knl-0021", "knl-0022", "knl-0036", "knl-0137", "knl-0138", "knl-0139", "knl-1234"]
    with open("node_list", "w") as f:
        for i, node in enumerate(nodelist_in):
            f.write(node + "\n")
            if i == 4:
                f.write(mynode + "\n")

    libE_specs = {"dedicated_mode": False}
    gresources = GlobalResources(libE_specs)
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

    os.remove("node_list")


def test_map_workerid_to_index():
    num_workers = 4
    num_rsets = 4

    zero_resource_list = []
    index_list = ResourceManager.get_index_list(num_workers, num_rsets, zero_resource_list)
    for workerID in range(1, num_workers + 1):
        index = index_list[workerID - 1]
        assert index == workerID - 1, "index incorrect. Received: " + str(index)

    zero_resource_list = [1]
    index_list = ResourceManager.get_index_list(num_workers, num_rsets, zero_resource_list)
    for workerID in range(2, num_workers + 1):
        index = index_list[workerID - 1]
        assert index == workerID - 2, "index incorrect. Received: " + str(index)

    zero_resource_list = [1, 2]
    index_list = ResourceManager.get_index_list(num_workers, num_rsets, zero_resource_list)
    for workerID in range(3, num_workers + 1):
        index = index_list[workerID - 1]
        assert index == workerID - 3, "index incorrect. Received: " + str(index)

    zero_resource_list = [1, 3]
    index_list = ResourceManager.get_index_list(num_workers, num_rsets, zero_resource_list)

    workerID = 2
    index = index_list[workerID - 1]
    assert index == 0, "index incorrect. Received: " + str(index)

    workerID = 4
    index = index_list[workerID - 1]
    assert index == 1, "index incorrect. Received: " + str(index)


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
    group_list, slot_list = ResourceManager.get_group_list(split_list)
    assert group_list == [1, 2, 3, 4, 5, 6, 7, 8]
    assert slot_list == [0, 0, 0, 0, 0, 0, 0, 0]

    # 4 resource sets with 2 nodes each ---------------------------------------
    split_list = [
        ["knl-0020", "knl-0021"],
        ["knl-0022", "knl-0036"],
        ["knl-0137", "knl-0138"],
        ["knl-0139", "knl-1234"],
    ]
    group_list, slot_list = ResourceManager.get_group_list(split_list)
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
    group_list, slot_list = ResourceManager.get_group_list(split_list)
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
    group_list, slot_list = ResourceManager.get_group_list(split_list)
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

    test_map_workerid_to_index()
    test_get_group_list()
    test_machinefile_from_resources()

    teardown_standalone_run()
