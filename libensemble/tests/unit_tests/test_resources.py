import os
import socket
from libensemble.resources import Resources, WorkerResources, ResourcesException


def setup_standalone_run():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = ""
    if os.path.isfile('worker_list'):
        os.remove('worker_list')


def teardown_standalone_run():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = ""
    if os.path.isfile('worker_list'):
        os.remove('worker_list')


def setup_function(function):
    print("setup_function    function:%s" % function.__name__)
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = ""
    # if os.environ['LIBE_RESOURCES_TEST_NODE_LIST']:
    #     del os.environ['LIBE_RESOURCES_TEST_NODE_LIST']
    # if os.environ['THIS_ENV_VARIABLE_IS_DEF_NOT_SET']:
    #     del os.environ['THIS_ENV_VARIABLE_IS_DEF_NOT_SET']
    if os.path.isfile('worker_list'):
        os.remove('worker_list')


def teardown_function(function):
    print("teardown_function    function:%s" % function.__name__)
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = ""
    # if os.environ['LIBE_RESOURCES_TEST_NODE_LIST']:
    #     del os.environ['LIBE_RESOURCES_TEST_NODE_LIST']
    # if os.environ['THIS_ENV_VARIABLE_IS_DEF_NOT_SET']:
    #     del os.environ['THIS_ENV_VARIABLE_IS_DEF_NOT_SET']
    if os.path.isfile('worker_list'):
        os.remove('worker_list')


# Tests ========================================================================================

# Tests for obtaining nodelist from environment variables

def test_slurm_nodelist_empty():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = ""
    exp_out = []  # empty
    nodelist = Resources.get_slurm_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_slurm_nodelist_single():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "knl-0056"
    exp_out = ["knl-0056"]
    nodelist = Resources.get_slurm_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_slurm_nodelist_knl_seq():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "knl-[0009-0012]"
    exp_out = ['knl-0009', 'knl-0010', 'knl-0011', 'knl-0012']
    nodelist = Resources.get_slurm_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_slurm_nodelist_bdw_seq():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "bdw-[0009-0012]"
    exp_out = ['bdw-0009', 'bdw-0010', 'bdw-0011', 'bdw-0012']
    nodelist = Resources.get_slurm_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_slurm_nodelist_groups():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "knl-[0020-0022,0137-0139,1234]"
    exp_out = ['knl-0020', 'knl-0021', 'knl-0022', 'knl-0137', 'knl-0138', 'knl-0139', 'knl-1234']
    nodelist = Resources.get_slurm_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_slurm_nodelist_groups_longprefix():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "super-[000020-000022,000137-000139,001234,023456-023458]"
    exp_out = ['super-000020', 'super-000021', 'super-000022', 'super-000137', 'super-000138', 'super-000139',
               'super-001234', 'super-023456', 'super-023457', 'super-023458']
    nodelist = Resources.get_slurm_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_slurm_nodelist_reverse_grp():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "knl-[0020-0022,0139-0137,1234]"
    exp_out = ['knl-0020', 'knl-0021', 'knl-0022', 'knl-0137', 'knl-0138', 'knl-0139', 'knl-1234']
    nodelist = Resources.get_slurm_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_cobalt_nodelist_empty():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = ""
    exp_out = []  # empty
    nodelist = Resources.get_cobalt_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_cobalt_nodelist_single():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "56"
    exp_out = ["56"]
    nodelist = Resources.get_cobalt_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_cobalt_nodelist_seq():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "9-12"
    exp_out = ['9', '10', '11', '12']
    nodelist = Resources.get_cobalt_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_cobalt_nodelist_groups():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "20-22,137-139,1234"
    exp_out = ['20', '21', '22', '137', '138', '139', '1234']
    nodelist = Resources.get_cobalt_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_cobalt_nodelist_reverse_grp():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "20-22,139-137,1234"
    exp_out = ['20', '21', '22', '137', '138', '139', '1234']
    nodelist = Resources.get_cobalt_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_lsf_nodelist_empty():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = ""
    exp_out = []  # empty
    nodelist = Resources.get_lsf_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_lsf_nodelist_single():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = 'batch5' + ' g06n02' * 42
    exp_out = ["g06n02"]
    nodelist = Resources.get_lsf_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_lsf_nodelist_seq():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = 'batch5' + ' g06n02' * 42 + ' h21n18' * 42
    exp_out = ['g06n02', 'h21n18']
    nodelist = Resources.get_lsf_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"

# These dont apply to the lsf lists as they are listed in full
# def test_lsf_nodelist_groups():
# def test_lsf_nodelist_reverse_grp():


# Tests Resources.get_global_nodelist (This requires above tests to work)
def test_get_global_nodelist_frm_slurm():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "knl-[0020-0022,0137-0139,1234]"
    exp_out = ['knl-0020', 'knl-0021', 'knl-0022', 'knl-0137', 'knl-0138', 'knl-0139', 'knl-1234']
    global_nodelist = Resources.get_global_nodelist(rundir=os.getcwd(),
                                                    nodelist_env_slurm="LIBE_RESOURCES_TEST_NODE_LIST",
                                                    nodelist_env_cobalt="THIS_ENV_VARIABLE_IS_DEF_NOT_SET")
    assert global_nodelist == exp_out, "global_nodelist returned does not match expected"


def test_get_global_nodelist_frm_cobalt():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "20-22,137-139,1234"
    exp_out = ['20', '21', '22', '137', '138', '139', '1234']
    global_nodelist = Resources.get_global_nodelist(rundir=os.getcwd(),
                                                    nodelist_env_slurm="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
                                                    nodelist_env_cobalt="LIBE_RESOURCES_TEST_NODE_LIST")
    assert global_nodelist == exp_out, "global_nodelist returned does not match expected"


def test_get_global_nodelist_frm_lsf():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = 'batch5' + ' g06n02' * 42 + ' h21n18' * 42
    exp_out = ['g06n02', 'h21n18']
    global_nodelist = Resources.get_global_nodelist(rundir=os.getcwd(),
                                                    nodelist_env_slurm="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
                                                    nodelist_env_cobalt="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
                                                    nodelist_env_lsf="LIBE_RESOURCES_TEST_NODE_LIST")
    assert global_nodelist == exp_out, "global_nodelist returned does not match expected"


def test_get_global_nodelist_standalone():
    mynode = socket.gethostname()
    global_nodelist = Resources.get_global_nodelist(rundir=os.getcwd(),
                                                    nodelist_env_slurm="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
                                                    nodelist_env_cobalt="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
                                                    nodelist_env_lsf="THIS_ENV_VARIABLE_IS_DEF_NOT_SET")
    assert global_nodelist == [mynode], "global_nodelist returned does not match expected"


def test_get_global_nodelist_frm_wrklst_file():
    # worker_list file should override env variables
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "20-22,137-139,1234"  # Should not be this
    exp_out = ['knl-0019', 'knl-0021', 'knl-0022', 'knl-0137', 'knl-0138', 'knl-0139', 'knl-2345']  # Should be this

    open('worker_list', 'w').close()
    try:
        _ = Resources.get_global_nodelist(rundir=os.getcwd())
    except ResourcesException as e:
        assert e.args[0] == 'Error. global_nodelist is empty'
    else:
        assert 0

    with open('worker_list', 'w') as f:
        for node in exp_out:
            f.write(node + '\n')

    # Do not specify env vars.
    global_nodelist1 = Resources.get_global_nodelist(rundir=os.getcwd())
    assert global_nodelist1 == exp_out, "global_nodelist returned does not match expected"

    # Specify env vars - should ignore
    global_nodelist2 = Resources.get_global_nodelist(rundir=os.getcwd(),
                                                     nodelist_env_slurm="THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
                                                     nodelist_env_cobalt="LIBE_RESOURCES_TEST_NODE_LIST")
    assert global_nodelist2 == exp_out, "global_nodelist returned does not match expected"
    os.remove('worker_list')


def test_remove_libE_nodes():
    mynode = socket.gethostname()
    exp_out = ['knl-0019', 'knl-0021', 'knl-0022', 'knl-0137', 'knl-0138', 'knl-0139', 'knl-2345']

    # Add at beginning
    nodes_in = [mynode] + exp_out
    nodes_out = Resources.remove_nodes(nodes_in, mynode)
    assert nodes_out == exp_out, "nodelist returned does not match expected"

    # Add twice in middle and at end
    nodes_in = []
    for i, node in enumerate(exp_out):
        nodes_in.append(node)
        if i == 1 or i == 4 or i == 6:
            nodes_in.append(mynode)
    nodes_out = Resources.remove_nodes(nodes_in, mynode)
    assert nodes_out == exp_out, "nodelist returned does not match expected"


def test_get_local_nodelist_central_mode():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "knl-[0020-0022,0036,0137-0139,1234]"
    resources = Resources(nodelist_env_slurm="LIBE_RESOURCES_TEST_NODE_LIST", central_mode=True)

    # Now mock up some more stuff - so consistent

    # Spoof current process as each worker and check nodelist.
    num_workers = 8
    exp_out = [['knl-0020'], ['knl-0021'], ['knl-0022'], ['knl-0036'], ['knl-0137'], ['knl-0138'], ['knl-0139'], ['knl-1234']]
    for wrk in range(num_workers):
        workerID = wrk + 1
        local_nodelist = WorkerResources.get_local_nodelist(num_workers, workerID, resources)
        assert local_nodelist == exp_out[wrk], "local_nodelist returned does not match expected"

    # Spoof current process as each worker and check nodelist.
    num_workers = 4
    exp_out = [['knl-0020', 'knl-0021'], ['knl-0022', 'knl-0036'], ['knl-0137', 'knl-0138'], ['knl-0139', 'knl-1234']]
    for wrk in range(num_workers):
        workerID = wrk + 1
        local_nodelist = WorkerResources.get_local_nodelist(num_workers, workerID, resources)
        assert local_nodelist == exp_out[wrk], "local_nodelist returned does not match expected"

    # Spoof current process as each worker and check nodelist.
    num_workers = 1
    exp_out = [['knl-0020', 'knl-0021', 'knl-0022', 'knl-0036', 'knl-0137', 'knl-0138', 'knl-0139', 'knl-1234']]
    for wrk in range(num_workers):
        workerID = wrk + 1
        local_nodelist = WorkerResources.get_local_nodelist(num_workers, workerID, resources)
        assert local_nodelist == exp_out[wrk], "local_nodelist returned does not match expected"

    # Test the best_split algorithm
    num_workers = 3
    exp_out = [['knl-0020', 'knl-0021', 'knl-0022'], ['knl-0036', 'knl-0137', 'knl-0138'], ['knl-0139', 'knl-1234']]
    for wrk in range(num_workers):
        workerID = wrk + 1
        local_nodelist = WorkerResources.get_local_nodelist(num_workers, workerID, resources)
        assert local_nodelist == exp_out[wrk], "local_nodelist returned does not match expected"


# The main tests are same as above - note for when fixtures set up
def test_get_local_nodelist_central_mode_remove_libE_proc():
    mynode = socket.gethostname()
    nodelist_in = ['knl-0020', 'knl-0021', 'knl-0022', 'knl-0036', 'knl-0137', 'knl-0138', 'knl-0139', 'knl-1234']
    with open('worker_list', 'w') as f:
        for i, node in enumerate(nodelist_in):
            f.write(node + '\n')
            if i == 3:
                f.write(mynode + '\n')

    resources = Resources(central_mode=True)

    # Now mock up some more stuff - so consistent

    # Spoof current process as each worker and check nodelist.
    num_workers = 8
    exp_out = [['knl-0020'], ['knl-0021'], ['knl-0022'], ['knl-0036'], ['knl-0137'], ['knl-0138'], ['knl-0139'], ['knl-1234']]
    for wrk in range(num_workers):
        workerID = wrk + 1
        local_nodelist = WorkerResources.get_local_nodelist(num_workers, workerID, resources)
        assert local_nodelist == exp_out[wrk], "local_nodelist returned does not match expected"

    # Spoof current process as each worker and check nodelist.
    num_workers = 4
    exp_out = [['knl-0020', 'knl-0021'], ['knl-0022', 'knl-0036'], ['knl-0137', 'knl-0138'], ['knl-0139', 'knl-1234']]
    for wrk in range(num_workers):
        workerID = wrk + 1
        local_nodelist = WorkerResources.get_local_nodelist(num_workers, workerID, resources)
        assert local_nodelist == exp_out[wrk], "local_nodelist returned does not match expected"

    # Spoof current process as each worker and check nodelist.
    num_workers = 1
    exp_out = [['knl-0020', 'knl-0021', 'knl-0022', 'knl-0036', 'knl-0137', 'knl-0138', 'knl-0139', 'knl-1234']]
    for wrk in range(num_workers):
        workerID = wrk + 1
        local_nodelist = WorkerResources.get_local_nodelist(num_workers, workerID, resources)
        assert local_nodelist == exp_out[wrk], "local_nodelist returned does not match expected"

    # Test the best_split algorithm
    num_workers = 3
    exp_out = [['knl-0020', 'knl-0021', 'knl-0022'], ['knl-0036', 'knl-0137', 'knl-0138'], ['knl-0139', 'knl-1234']]
    for wrk in range(num_workers):
        workerID = wrk + 1
        local_nodelist = WorkerResources.get_local_nodelist(num_workers, workerID, resources)
        assert local_nodelist == exp_out[wrk], "local_nodelist returned does not match expected"

    os.remove('worker_list')


def test_get_local_nodelist_distrib_mode_host_not_in_list():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "knl-[0020-0022,0036,0137-0139,1234]"
    resources = Resources(nodelist_env_slurm="LIBE_RESOURCES_TEST_NODE_LIST", central_mode=False)

    # Spoof current process as each worker and check nodelist.
    num_workers = 4
    exp_out = ['knl-0022', 'knl-0036']

    # Test running distributed mode without current host in list.
    workerID = 2
    local_nodelist = WorkerResources.get_local_nodelist(num_workers, workerID, resources)

    # Now this should work
    assert local_nodelist == exp_out, "local_nodelist returned does not match expected"
    # try:
    #     local_nodelist = WorkerResources.get_local_nodelist(num_workers, workerID, resources)
    # except:
    #     assert 1
    # else:
    #     assert 0


def test_get_local_nodelist_distrib_mode():
    mynode = socket.gethostname()
    # nodelist_in = ['knl-0020', 'knl-0021', 'knl-0022', 'knl-0036', 'knl-0137', 'knl-0138', 'knl-0139', 'knl-1234']
    nodelist_in = ['knl-0020', 'knl-0021', 'knl-0022', 'knl-0036', 'knl-0137', 'knl-0138', 'knl-0139']
    with open('worker_list', 'w') as f:
        for i, node in enumerate(nodelist_in):
            f.write(node + '\n')
            if i == 3:
                f.write(mynode + '\n')

    resources = Resources(central_mode=False)

    # Spoof current process as each worker and check nodelist.
    num_workers = 8

    # Test workerID not in local_nodelist [update: This should now work - check removed]
    # workerID = 4
    # try:
    #     local_nodelist = WorkerResources.get_local_nodelist(num_workers, workerID, resources)
    # except:
    #     assert 1
    # else:
    #     assert 0

    workerID = 5
    exp_out = [mynode]
    local_nodelist = WorkerResources.get_local_nodelist(num_workers, workerID, resources)
    assert local_nodelist == exp_out, "local_nodelist returned does not match expected"

    num_workers = 1
    workerID = 1
    exp_out = ['knl-0020', 'knl-0021', 'knl-0022', 'knl-0036', mynode, 'knl-0137', 'knl-0138', 'knl-0139']
    local_nodelist = WorkerResources.get_local_nodelist(num_workers, workerID, resources)
    assert local_nodelist == exp_out, "local_nodelist returned does not match expected"

    num_workers = 4
    workerID = 3
    exp_out = [mynode, 'knl-0137']
    local_nodelist = WorkerResources.get_local_nodelist(num_workers, workerID, resources)
    assert local_nodelist == exp_out, "local_nodelist returned does not match expected"

    # Sub-node workers
    num_workers = 16

    workerID = 9
    exp_out = [mynode]
    local_nodelist = WorkerResources.get_local_nodelist(num_workers, workerID, resources)
    assert local_nodelist == exp_out, "local_nodelist returned does not match expected"

    workerID = 10
    exp_out = [mynode]

    local_nodelist = WorkerResources.get_local_nodelist(num_workers, workerID, resources)
    assert local_nodelist == exp_out, "local_nodelist returned does not match expected"
    os.remove('worker_list')


def test_get_local_nodelist_distrib_mode_uneven_split():
    mynode = socket.gethostname()
    nodelist_in = ['knl-0020', 'knl-0021', 'knl-0022', 'knl-0036', 'knl-0137', 'knl-0138', 'knl-0139', 'knl-1234']
    with open('worker_list', 'w') as f:
        for i, node in enumerate(nodelist_in):
            f.write(node + '\n')
            if i == 4:
                f.write(mynode + '\n')

    resources = Resources(central_mode=False)
    num_workers = 2

    # May not be at head of list - should perhaps be warning or enforced
    workerID = 2
    exp_out = ['knl-0137', mynode, 'knl-0138', 'knl-0139']
    local_nodelist = WorkerResources.get_local_nodelist(num_workers, workerID, resources)
    assert local_nodelist == exp_out, "local_nodelist returned does not match expected"
    os.remove('worker_list')


class Fake_comm():
    def __init__(self, nworkers):
        self.num_workers = nworkers

    def get_num_workers(self):
        return self.num_workers


def test_worker_resources():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "knl-[0020-0022,0036,0137-0139,1234]"
    resources = Resources(nodelist_env_slurm="LIBE_RESOURCES_TEST_NODE_LIST", central_mode=True)

    # One worker per node
    exp_nodelist1 = [['knl-0020'], ['knl-0021'], ['knl-0022'], ['knl-0036'], ['knl-0137'], ['knl-0138'], ['knl-0139'], ['knl-1234']]
    num_workers = 8
    comm = Fake_comm(num_workers)
    for wrk in range(num_workers):
        workerID = wrk + 1
        worker = WorkerResources(workerID, comm, resources)
        assert worker.num_workers == 8, 'worker.num_workers does not match'
        assert worker.workerID == workerID, 'worker.workerID does not match'
        assert worker.local_nodelist == exp_nodelist1[wrk], 'worker.local_nodelist does not match'
        assert worker.local_node_count == 1, 'worker.local_node_count does not match'
        assert worker.workers_per_node == 1, 'worker.workers_per_node does not match'

    # Multiple nodes per worker
    exp_nodelist2 = [['knl-0020', 'knl-0021', 'knl-0022', 'knl-0036'], ['knl-0137', 'knl-0138', 'knl-0139', 'knl-1234']]
    num_workers = 2
    comm2 = Fake_comm(num_workers)
    for wrk in range(num_workers):
        workerID = wrk + 1
        worker = WorkerResources(workerID, comm2, resources)
        assert worker.num_workers == 2, 'worker.num_workers does not match'
        assert worker.workerID == workerID, 'worker.workerID does not match'
        assert worker.local_nodelist == exp_nodelist2[wrk], 'worker.local_nodelist does not match'
        assert worker.local_node_count == 4, 'worker.local_node_count does not match'
        assert worker.workers_per_node == 1, 'worker.workers_per_node does not match'

    # Multiple workers per node
    num_workers = 16
    comm3 = Fake_comm(num_workers)
    for wrk in range(num_workers):
        workerID = wrk + 1
        worker = WorkerResources(workerID, comm3, resources)
        assert worker.num_workers == 16, 'worker.num_workers does not match'
        assert worker.workerID == workerID, 'worker.workerID does not match'
        assert worker.local_nodelist == exp_nodelist1[wrk//2], 'worker.local_nodelist does not match'
        assert worker.local_node_count == 1, 'worker.local_node_count does not match'
        assert worker.workers_per_node == 2, 'worker.workers_per_node does not match'


if __name__ == "__main__":
    setup_standalone_run()

    test_slurm_nodelist_empty()
    test_slurm_nodelist_single()
    test_slurm_nodelist_knl_seq()
    test_slurm_nodelist_bdw_seq()
    test_slurm_nodelist_groups()
    test_slurm_nodelist_groups_longprefix()
    test_slurm_nodelist_reverse_grp()

    test_cobalt_nodelist_empty()
    test_cobalt_nodelist_single()
    test_cobalt_nodelist_seq()
    test_cobalt_nodelist_groups()
    test_cobalt_nodelist_reverse_grp()

    test_lsf_nodelist_empty()
    test_lsf_nodelist_single()
    test_lsf_nodelist_seq()

    test_get_global_nodelist_frm_slurm()
    test_get_global_nodelist_frm_cobalt()
    test_get_global_nodelist_frm_lsf()
    test_get_global_nodelist_standalone()

    test_get_global_nodelist_frm_wrklst_file()
    test_remove_libE_nodes()
    test_get_local_nodelist_central_mode()
    test_get_local_nodelist_central_mode_remove_libE_proc()
    test_get_local_nodelist_distrib_mode_host_not_in_list()
    test_get_local_nodelist_distrib_mode()
    test_get_local_nodelist_distrib_mode_uneven_split()

    test_worker_resources()
    teardown_standalone_run()
