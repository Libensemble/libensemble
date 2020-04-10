import os
import shutil
import numpy as np
from libensemble.libE_worker import Worker
from libensemble.utils.loc_stack import LocationStack


def test_range_single_element():
    """Single H_row labeling """

    work = {'H_fields': ['x', 'num_nodes', 'ranks_per_node'],
            'libE_info': {'H_rows': np.array([5]), 'workerID': 1}}
    assert Worker._extract_H_ranges(work) == '5', \
        'Failed to correctly parse single H row'


def test_range_two_separate_elements():
    """Multiple H_rows, non-sequential"""

    work = {'H_fields': ['x', 'num_nodes', 'ranks_per_node'],
            'libE_info': {'H_rows': np.array([2, 8]), 'workerID': 1}}
    assert Worker._extract_H_ranges(work) == '2_8', \
        'Failed to correctly parse nonsequential H rows'


def test_range_two_ranges():
    """Multiple sequences of H_rows"""

    work = {'H_fields': ['x', 'num_nodes', 'ranks_per_node'],
            'libE_info': {'H_rows': np.array([0, 1, 2, 3, 7, 8]),
                          'workerID': 1}}
    assert Worker._extract_H_ranges(work) == '0-3_7-8', \
        'Failed to correctly parse multiple H ranges'


def test_range_mixes():
    """Mix of single rows and sequences of H_rows"""

    work = {'H_fields': ['x', 'num_nodes', 'ranks_per_node'],
            'libE_info': {'H_rows': np.array([2, 3, 4, 6, 8, 9, 11, 14]),
                          'workerID': 1}}
    assert Worker._extract_H_ranges(work) == '2-4_6_8-9_11_14', \
        'Failed to correctly parse H row single elements and ranges.'


def test_copy_back():
    """ When workers conclude their work, workers have the option of copying
    back their work into a directory created by the manager."""

    class FakeWorker:
        """ Enough information to test _copy_back() """
        def __init__(self, libE_specs, prefix, startdir):
            self.libE_specs = libE_specs
            self.prefix = prefix
            self.startdir = startdir

    # Using directories and files from previous test
    libE_specs = {'make_sim_dirs': True, 'sim_dir_path': './input', 'sim_dir_copy_back': True}
    prefix = './calc'
    copybackdir = './calc_back'
    startdir = '.'

    # Normally created by manager
    os.makedirs(copybackdir, exist_ok=True)

    fake_worker = FakeWorker(libE_specs, prefix, startdir)
    Worker._copy_back(fake_worker)
    assert 'file' in os.listdir(copybackdir), \
        'File not copied back to starting dir'

    for dir in [prefix, copybackdir, 'input']:
        shutil.rmtree(dir)


if __name__ == '__main__':
    test_range_single_element()
    test_range_two_separate_elements()
    test_range_two_ranges()
    test_range_mixes()
    test_copy_back()
