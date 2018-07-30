import os
from libensemble.resources import Resources


def setup_function(function):
    print ("setup_function    function:%s" % function.__name__)
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = ""
    #if os.environ['LIBE_RESOURCES_TEST_NODE_LIST']:
        #del os.environ['LIBE_RESOURCES_TEST_NODE_LIST']
    #if os.environ['THIS_ENV_VARIABLE_IS_DEF_NOT_SET']:
        #del os.environ['THIS_ENV_VARIABLE_IS_DEF_NOT_SET']
    if os.path.isfile('worker_list'):
        os.remove('worker_list')

def teardown_function(function):
    print ("teardown_function    function:%s" % function.__name__)
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = ""
    #if os.environ['LIBE_RESOURCES_TEST_NODE_LIST']:
        #del os.environ['LIBE_RESOURCES_TEST_NODE_LIST']
    #if os.environ['THIS_ENV_VARIABLE_IS_DEF_NOT_SET']:
        #del os.environ['THIS_ENV_VARIABLE_IS_DEF_NOT_SET']
    if os.path.isfile('worker_list'):
        os.remove('worker_list')
    

# Tests for obtaining nodelist from environment variables

def test_slurm_nodelist_empty():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = ""
    exp_out = [] #empty
    nodelist = Resources.get_slurm_nodelist(node_list_env = "LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned is does not match expected"
    print(nodelist)

def test_slurm_nodelist_single():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "knl-0056"
    exp_out = ["knl-0056"]
    nodelist = Resources.get_slurm_nodelist(node_list_env = "LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned is does not match expected"


def test_slurm_nodelist_knl_seq():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "knl-[0009-0012]"
    exp_out = ['knl-0009', 'knl-0010', 'knl-0011', 'knl-0012']
    nodelist = Resources.get_slurm_nodelist(node_list_env = "LIBE_RESOURCES_TEST_NODE_LIST")   
    assert nodelist == exp_out, "Nodelist returned is does not match expected"


def test_slurm_nodelist_bdw_seq():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "bdw-[0009-0012]"
    exp_out = ['bdw-0009', 'bdw-0010', 'bdw-0011', 'bdw-0012']
    nodelist = Resources.get_slurm_nodelist(node_list_env = "LIBE_RESOURCES_TEST_NODE_LIST")  
    assert nodelist == exp_out, "Nodelist returned is does not match expected"
    

def test_slurm_nodelist_groups():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "knl-[0020-0022,0137-0139,1234]"
    exp_out = ['knl-0020', 'knl-0021', 'knl-0022', 'knl-0137', 'knl-0138', 'knl-0139', 'knl-1234']
    nodelist = Resources.get_slurm_nodelist(node_list_env = "LIBE_RESOURCES_TEST_NODE_LIST")   
    assert nodelist == exp_out, "Nodelist returned is does not match expected"    


def test_slurm_nodelist_groups_longprefix():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "super-[000020-000022,000137-000139,001234,023456-023458]"
    exp_out = ['super-000020', 'super-000021', 'super-000022', 'super-000137', 'super-000138', 'super-000139',
               'super-001234', 'super-023456', 'super-023457', 'super-023458']
    nodelist = Resources.get_slurm_nodelist(node_list_env = "LIBE_RESOURCES_TEST_NODE_LIST")   
    assert nodelist == exp_out, "Nodelist returned is does not match expected"    


def test_cobalt_nodelist_empty():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = ""
    exp_out = [] #empty
    nodelist = Resources.get_cobalt_nodelist(node_list_env = "LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned is does not match expected"
    
    
def test_cobalt_nodelist_single():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "56"
    exp_out = ["nid56"]
    nodelist = Resources.get_cobalt_nodelist(node_list_env = "LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned is does not match expected"


def test_cobalt_nodelist_seq():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "9-12"
    exp_out = ['nid09', 'nid10', 'nid11', 'nid12']
    nodelist = Resources.get_cobalt_nodelist(node_list_env = "LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned is does not match expected"


def test_cobalt_nodelist_groups():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "20-22,137-139,1234"
    
    #exp_out = ['nid20', 'nid21', 'nid22', 'nid137', 'nid138', 'nid139', 'nid1234'] 
    #Check ordering....
    exp_out = ['nid1234', 'nid137', 'nid138', 'nid139', 'nid20', 'nid21', 'nid22'] 
    nodelist = Resources.get_cobalt_nodelist(node_list_env = "LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned is does not match expected"
    #print(exp_out)
    #print(nodelist) 

#To check
#def test_cobalt_nodelist_groups_longprefix():
    #pass


# Tests Resources.get_global_nodelist (This requires above tests to work)

def test_get_global_nodelist_frm_slurm():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "knl-[0020-0022,0137-0139,1234]"
    exp_out = ['knl-0020', 'knl-0021', 'knl-0022', 'knl-0137', 'knl-0138', 'knl-0139', 'knl-1234']
    global_nodelist = Resources.get_global_nodelist(rundir=os.getcwd(), central_mode=False,
                                                    nodelist_env_slurm = "LIBE_RESOURCES_TEST_NODE_LIST",
                                                    nodelist_env_cobalt = "THIS_ENV_VARIABLE_IS_DEF_NOT_SET")   
    assert global_nodelist == exp_out, "global_nodelist returned is does not match expected"


def test_get_global_nodelist_frm_cobalt():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "20-22,137-139,1234"
    exp_out = ['nid1234', 'nid137', 'nid138', 'nid139', 'nid20', 'nid21', 'nid22'] 
    global_nodelist = Resources.get_global_nodelist(rundir=os.getcwd(), central_mode=False,
                                                    nodelist_env_slurm = "THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
                                                    nodelist_env_cobalt = "LIBE_RESOURCES_TEST_NODE_LIST")   
    assert global_nodelist == exp_out, "global_nodelist returned is does not match expected"
    

def test_get_global_nodelist_frm_wrklst_file():
    # worker_list file should override env variables
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "20-22,137-139,1234" # Should not be this
    exp_out = ['knl-0019', 'knl-0021', 'knl-0022', 'knl-0137', 'knl-0138', 'knl-0139', 'knl-2345'] # Should be this
    
    #Try empty (really want to start testing error messages - should be "Error. global_nodelist is empty"
    open('worker_list','w').close()
    try:
        global_nodelist = Resources.get_global_nodelist(rundir=os.getcwd(), central_mode=False)
    except: 
        assert 1
    else:
        assert 0   
    
    with open('worker_list','w') as f:
        for node in exp_out:
            f.write(node + '\n')
            
    # Do not specify env vars.        
    global_nodelist = Resources.get_global_nodelist(rundir=os.getcwd(), central_mode=False)
    assert global_nodelist == exp_out, "global_nodelist returned is does not match expected"
    
    # Specify env vars - should ignore
    global_nodelist2 = Resources.get_global_nodelist(rundir=os.getcwd(), central_mode=False,
                                                    nodelist_env_slurm = "THIS_ENV_VARIABLE_IS_DEF_NOT_SET",
                                                    nodelist_env_cobalt = "LIBE_RESOURCES_TEST_NODE_LIST")       
    assert global_nodelist2 == exp_out, "global_nodelist returned is does not match expected"
    #print(global_nodelist2)
    #os.remove('worker_list')
    

if __name__ == "__main__":
    #test_slurm_nodelist_empty
    #test_slurm_nodelist_single()
    #test_slurm_nodelist_seq()
    #test_slurm_nodelist_bdw_seq()
    #test_slurm_nodelist_groups()
    #test_slurm_nodelist_groups_longprefix()
    #test_cobalt_nodelist_empty
    #test_cobalt_nodelist_single()
    #test_cobalt_nodelist_seq()
    #test_cobalt_nodelist_groups()
    #test_get_global_nodelist_frm_slurm()
    #test_get_global_nodelist_frm_cobalt()
    test_get_global_nodelist_frm_wrklst_file()
    
    
    
