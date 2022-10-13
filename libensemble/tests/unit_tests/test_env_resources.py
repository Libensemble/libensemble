import os
from libensemble.resources.env_resources import EnvResources


def setup_standalone_run():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = ""


def teardown_standalone_run():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = ""


def setup_function(function):
    print(f"setup_function    function:{function.__name__}")
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = ""


def teardown_function(function):
    print(f"teardown_function    function:{function.__name__}")
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = ""


# Tests ========================================================================================

# Tests for obtaining nodelist from environment variables


def test_slurm_nodelist_empty():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = ""
    exp_out = []  # empty
    nodelist = EnvResources.get_slurm_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_slurm_nodelist_single():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "knl-0056"
    exp_out = ["knl-0056"]
    nodelist = EnvResources.get_slurm_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_slurm_nodelist_single_nodash():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "nid00056"
    exp_out = ["nid00056"]
    nodelist = EnvResources.get_slurm_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_slurm_nodelist_knl_seq():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "knl-[0009-0012]"
    exp_out = ["knl-0009", "knl-0010", "knl-0011", "knl-0012"]
    nodelist = EnvResources.get_slurm_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_slurm_nodelist_bdw_seq():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "bdw-[0009-0012]"
    exp_out = ["bdw-0009", "bdw-0010", "bdw-0011", "bdw-0012"]
    nodelist = EnvResources.get_slurm_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_slurm_nodelist_groups():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "knl-[0020-0022,0137-0139,1234]"
    exp_out = ["knl-0020", "knl-0021", "knl-0022", "knl-0137", "knl-0138", "knl-0139", "knl-1234"]
    nodelist = EnvResources.get_slurm_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_slurm_nodelist_groups_partitions():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "bdw-[0254,0384,0565-0568],bdwd-[0004,0009]"
    exp_out = ["bdw-0254", "bdw-0384", "bdw-0565", "bdw-0566", "bdw-0567", "bdw-0568", "bdwd-0004", "bdwd-0009"]
    nodelist = EnvResources.get_slurm_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_slurm_nodelist_groups_nodash():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "nid0[0020-0022,0137-0139,1234]"
    exp_out = ["nid00020", "nid00021", "nid00022", "nid00137", "nid00138", "nid00139", "nid01234"]
    nodelist = EnvResources.get_slurm_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_slurm_nodelist_groups_longprefix():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "super-[000020-000022,000137-000139,001234,023456-023458]"
    exp_out = [
        "super-000020",
        "super-000021",
        "super-000022",
        "super-000137",
        "super-000138",
        "super-000139",
        "super-001234",
        "super-023456",
        "super-023457",
        "super-023458",
    ]
    nodelist = EnvResources.get_slurm_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_slurm_nodelist_reverse_grp():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "knl-[0020-0022,0139-0137,1234]"
    exp_out = ["knl-0020", "knl-0021", "knl-0022", "knl-0137", "knl-0138", "knl-0139", "knl-1234"]
    nodelist = EnvResources.get_slurm_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_cobalt_nodelist_empty():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = ""
    exp_out = []  # empty
    nodelist = EnvResources.get_cobalt_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_cobalt_nodelist_single():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "56"
    exp_out = ["56"]
    nodelist = EnvResources.get_cobalt_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_cobalt_nodelist_seq():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "9-12"
    exp_out = ["9", "10", "11", "12"]
    nodelist = EnvResources.get_cobalt_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_cobalt_nodelist_groups():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "20-22,137-139,1234"
    exp_out = ["20", "21", "22", "137", "138", "139", "1234"]
    nodelist = EnvResources.get_cobalt_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_cobalt_nodelist_reverse_grp():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "20-22,139-137,1234"
    exp_out = ["20", "21", "22", "137", "138", "139", "1234"]
    nodelist = EnvResources.get_cobalt_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_lsf_nodelist_empty():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = ""
    exp_out = []  # empty
    nodelist = EnvResources.get_lsf_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_lsf_nodelist_single():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "batch5" + " g06n02" * 42
    exp_out = ["g06n02"]
    nodelist = EnvResources.get_lsf_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_lsf_nodelist_seq():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "batch5" + " g06n02" * 42 + " h21n18" * 42
    exp_out = ["g06n02", "h21n18"]
    nodelist = EnvResources.get_lsf_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_pbs_nodelist_nofile():
    nodefile = "./pbs_nofile"
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = nodefile
    exp_out = []
    nodelist = EnvResources.get_pbs_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned should be empty if nodefile inaccessible"


def test_pbs_nodelist_empty():
    nodefile = "pbs_empty"
    open(nodefile, "w").close()

    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = nodefile
    exp_out = []
    nodelist = EnvResources.get_pbs_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned should be empty if nodefile empty"
    os.remove(nodefile)


def test_pbs_nodelist_single():
    nodefile = "pbs_single"
    in_lines = ["edtb-01.mcp.alcf.anl.gov\n", "edtb-01.mcp.alcf.anl.gov\n", "edtb-01.mcp.alcf.anl.gov\n"]
    with open(nodefile, "w") as f:
        for line in in_lines:
            f.write(line)

    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = nodefile
    exp_out = ["edtb-01.mcp.alcf.anl.gov"]
    nodelist = EnvResources.get_pbs_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"
    os.remove(nodefile)


def test_pbs_nodelist_seq():
    nodefile = "pbs_seq"
    in_lines = ["edtb-01.mcp.alcf.anl.gov\n", "edtb-02.mcp.alcf.anl.gov\n"]
    with open(nodefile, "w") as f:
        for line in in_lines:
            f.write(line)

    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = nodefile
    exp_out = ["edtb-01.mcp.alcf.anl.gov", "edtb-02.mcp.alcf.anl.gov"]
    nodelist = EnvResources.get_pbs_nodelist(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"
    os.remove(nodefile)


# These dont apply to the lsf lists as they are listed in full
# def test_lsf_nodelist_groups():
# def test_lsf_nodelist_reverse_grp():


def test_lsf_nodelist_shortform_empty():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = ""
    exp_out = []  # empty
    nodelist = EnvResources.get_lsf_nodelist_frm_shortform(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_lsf_nodelist_shortform_single():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "batch5 1 g06n02 42"
    exp_out = ["g06n02"]
    nodelist = EnvResources.get_lsf_nodelist_frm_shortform(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_lsf_nodelist_shortform_seq():
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "batch5 1 g06n02 42 h21n18 42"
    exp_out = ["g06n02", "h21n18"]
    nodelist = EnvResources.get_lsf_nodelist_frm_shortform(node_list_env="LIBE_RESOURCES_TEST_NODE_LIST")
    assert nodelist == exp_out, "Nodelist returned does not match expected"


def test_shortnames_nochange_slurm():
    env_resources = EnvResources()
    # Test Slurm abbrev
    exp_names = ["knl-0019", "knl-0021", "knl-0022", "knl-0137", "knl-0138", "knl-0139", "knl-2345"]
    env_resources.scheduler = "Slurm"
    abbrev_names = env_resources.shortnames(exp_names)
    assert abbrev_names == exp_names, "Abbreviated names returned do not match expected"
    del env_resources


def test_shortnames_slurm():
    env_resources = EnvResources()
    # Test Slurm abbrev
    exp_names = ["knl-0019", "knl-0021", "knl-0022"]
    full_names = ["knl-0019.some.suffix", "knl-0021.some.suffix", "knl-0022.diff_suffix"]
    env_resources.scheduler = "Slurm"
    abbrev_names = env_resources.shortnames(full_names)
    assert abbrev_names == exp_names, "Abbreviated names returned do not match expected"
    del env_resources


def test_shortnames_nochange_cobalt():
    env_resources = EnvResources()
    # Test Cobalt abbrev
    exp_names = ["21", "22", "137", "138", "1234", "11234"]
    env_resources.scheduler = "Cobalt"
    abbrev_names = env_resources.shortnames(exp_names)
    assert abbrev_names == exp_names, "Abbreviated names returned do not match expected"
    del env_resources


def test_shortnames_cobalt():
    env_resources = EnvResources()
    # Test Cobalt abbrev
    exp_names = ["20", "21", "22", "137", "138", "1234", "11234"]
    full_names = ["nid00020", "nid00021", "nid00022", "nid00137", "nid00138", "nid01234", "nid11234"]
    env_resources.scheduler = "Cobalt"
    abbrev_names = env_resources.shortnames(full_names)
    assert abbrev_names == exp_names, "Abbreviated names returned do not match expected"
    del env_resources


if __name__ == "__main__":
    setup_standalone_run()

    test_slurm_nodelist_empty()
    test_slurm_nodelist_single()
    test_slurm_nodelist_single_nodash()
    test_slurm_nodelist_knl_seq()
    test_slurm_nodelist_bdw_seq()
    test_slurm_nodelist_groups()
    test_slurm_nodelist_groups_partitions()
    test_slurm_nodelist_groups_nodash()
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

    test_pbs_nodelist_nofile()
    test_pbs_nodelist_empty()
    test_pbs_nodelist_single()
    test_pbs_nodelist_seq()

    test_lsf_nodelist_shortform_empty()
    test_lsf_nodelist_shortform_single()
    test_lsf_nodelist_shortform_seq()

    test_shortnames_nochange_slurm()
    test_shortnames_slurm()
    test_shortnames_nochange_cobalt()
    test_shortnames_cobalt()

    teardown_standalone_run()
