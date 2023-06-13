import os
import shutil
from pathlib import Path

import numpy as np

from libensemble.utils.loc_stack import LocationStack
from libensemble.utils.misc import extract_H_ranges
from libensemble.utils.output_directory import EnsembleDirectory


def test_range_single_element():
    """Single H_row labeling"""
    work = {"H_fields": ["x", "num_nodes", "procs_per_node"], "libE_info": {"H_rows": np.array([5]), "workerID": 1}}
    assert extract_H_ranges(work) == "5", "Failed to correctly parse single H row"


def test_range_two_separate_elements():
    """Multiple H_rows, non-sequential"""
    work = {"H_fields": ["x", "num_nodes", "procs_per_node"], "libE_info": {"H_rows": np.array([2, 8]), "workerID": 1}}
    assert extract_H_ranges(work) == "2_8", "Failed to correctly parse nonsequential H rows"


def test_range_two_ranges():
    """Multiple sequences of H_rows"""
    work = {
        "H_fields": ["x", "num_nodes", "procs_per_node"],
        "libE_info": {"H_rows": np.array([0, 1, 2, 3, 7, 8]), "workerID": 1},
    }
    assert extract_H_ranges(work) == "0-3_7-8", "Failed to correctly parse multiple H ranges"


def test_range_mixes():
    """Mix of single rows and sequences of H_rows"""
    work = {
        "H_fields": ["x", "num_nodes", "procs_per_node"],
        "libE_info": {"H_rows": np.array([2, 3, 4, 6, 8, 9, 11, 14]), "workerID": 1},
    }
    assert extract_H_ranges(work) == "2-4_6_8-9_11_14", "Failed to correctly parse H row single elements and ranges."


def test_copy_back(tmp_path):
    """When workers conclude their work, workers have the option of copying
    back their work into a directory created by the manager."""

    inputdir = tmp_path / "calc"
    copybackdir = "./calc_back"
    inputfile = tmp_path / "calc/file"

    for dire in [inputdir, copybackdir]:
        os.makedirs(dire, exist_ok=True)

    libE_specs = {"sim_dirs_make": True, "ensemble_dir_path": inputdir, "ensemble_copy_back": True}

    ls = LocationStack()
    ls.register_loc("test", inputfile)
    ed = EnsembleDirectory(libE_specs, ls)
    ed.copy_back()
    assert "file" in os.listdir(copybackdir), "File not copied back to starting dire"

    for dire in [inputdir, copybackdir]:
        shutil.rmtree(dire)

    # If copyback directory in starting location, test contents copied back to directory suffixed with _back
    inputdir = "./calc"
    copybackdir = "./calc_back"
    inputfile = "./calc/file"

    for dire in [inputdir, copybackdir]:
        os.makedirs(dire, exist_ok=True)

    libE_specs = {"sim_dirs_make": True, "ensemble_dir_path": inputdir, "ensemble_copy_back": True}

    ls = LocationStack()
    ls.register_loc("test", Path(inputfile))
    ed = EnsembleDirectory(libE_specs, ls)
    ed.copy_back()
    assert "file" in os.listdir(copybackdir), "File not copied back to starting dire"

    for dire in [inputdir, copybackdir]:
        shutil.rmtree(dire)


def test_worker_dirs_but_no_sim_dirs(tmp_path):
    """Test Worker._make_calc_dir() directory structure without sim_dirs"""
    inputdir = tmp_path / "calc"
    inputfile = tmp_path / "calc/file"
    ensemble_dir = tmp_path / "test_ens"

    for dire in [inputdir, inputfile, ensemble_dir]:
        os.makedirs(dire, exist_ok=True)

    libE_specs = {"ensemble_dir_path": ensemble_dir, "use_worker_dirs": True, "sim_input_dir": inputdir}

    ls = LocationStack()
    ed = EnsembleDirectory(libE_specs, ls)
    for i in range(4):  # Should work at any range
        EnsembleDirectory._make_calc_dir(ed, 1, 1, "sim", ls)

    assert "worker1" in os.listdir(ensemble_dir)
    assert "file" in os.listdir(os.path.join(ensemble_dir, "worker1"))

    for dire in [inputdir, ensemble_dir]:
        shutil.rmtree(dire)


def test_loc_stack_FileExists_exceptions(tmp_path):
    inputdir = tmp_path / "calc"
    copyfile = tmp_path / "calc/copy"
    symlinkfile = tmp_path / "calc/symlink"
    ensemble_dir = tmp_path / "test_ens"

    for dire in [inputdir, copyfile, symlinkfile]:
        os.makedirs(dire, exist_ok=True)

    # Testing loc_stack continuing on FileExistsError when not using sim_dirs
    libE_specs = {
        "sim_dirs_make": False,
        "ensemble_dir_path": ensemble_dir,
        "use_worker_dirs": True,
        "sim_dir_copy_files": [copyfile],
        "sim_dir_symlink_files": [symlinkfile],
    }

    ls = LocationStack()
    ed = EnsembleDirectory(libE_specs=libE_specs, loc_stack=ls)
    for i in range(2):  # Should work at any range
        EnsembleDirectory._make_calc_dir(ed, 1, "1", "sim", ls)

    assert len(os.listdir(ensemble_dir)) == 1, "Should only be a single worker file in ensemble"
    assert "worker1" in os.listdir(ensemble_dir), "Only directory should be worker1"
    assert all(
        [i in os.listdir(os.path.join(ensemble_dir, "worker1")) for i in ["copy", "symlink"]]
    ), "Files to copy and symlink not found in worker directory."

    # Testing loc_stack exception raising when sim_dir re-used - copy
    libE_specs = {
        "sim_dirs_make": True,
        "ensemble_dir_path": ensemble_dir,
        "use_worker_dirs": True,
        "sim_dir_copy_files": [copyfile],
    }

    flag = 1
    ed = EnsembleDirectory(libE_specs=libE_specs, loc_stack=ls)
    EnsembleDirectory._make_calc_dir(ed, 1, "1", "sim", ls)
    try:
        EnsembleDirectory._make_calc_dir(ed, 1, "1", "sim", ls)
    except FileExistsError:
        flag = 0
    assert flag == 0

    # Testing loc_stack exception raising when sim_dir re-used - symlink
    libE_specs = {
        "sim_dirs_make": True,
        "ensemble_dir_path": ensemble_dir,
        "use_worker_dirs": True,
        "sim_dir_symlink_files": [symlinkfile],
    }

    flag = 1
    ed = EnsembleDirectory(libE_specs=libE_specs, loc_stack=ls)
    EnsembleDirectory._make_calc_dir(ed, 1, "2", "sim", ls)
    try:
        EnsembleDirectory._make_calc_dir(ed, 1, "2", "sim", ls)
    except FileExistsError:
        flag = 0
    assert flag == 0

    for dire in [inputdir, ensemble_dir]:
        shutil.rmtree(dire)


def test_workflow_dir_copyback(tmp_path):
    """When workers conclude their work, workers have the option of copying
    back their work into the workflow directory."""

    inputdir = tmp_path / "calc"
    inputfile = tmp_path / "calc/file"

    for dire in [inputdir, inputfile]:
        os.makedirs(dire, exist_ok=True)

    libE_specs = {
        "sim_dirs_make": True,
        "ensemble_dir_path": tmp_path,
        "ensemble_copy_back": True,
        "use_workflow_dir": True,
        "workflow_dir_path": "./workflow_intermediate_copyback/fake_workflow",
    }

    ls = LocationStack()
    ls.register_loc("test", inputfile)
    ed = EnsembleDirectory(libE_specs, ls)
    copybackdir = ed.copybackdir

    assert "fake_workflow" in str(copybackdir), "workflow_dir wasn't considered as destination for copyback"

    ed.copy_back()
    assert "file" in os.listdir(copybackdir), "File not copied back to starting dire"

    for dire in [inputdir, copybackdir]:
        shutil.rmtree(dire)


if __name__ == "__main__":
    test_range_single_element()
    test_range_two_separate_elements()
    test_range_two_ranges()
    test_range_mixes()
    test_copy_back()
    test_worker_dirs_but_no_sim_dirs()
    test_loc_stack_FileExists_exceptions()
    test_workflow_dir_copyback()
