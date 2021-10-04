import sys
import pytest
import numpy as np
from libensemble.resources.scheduler import (ResourceScheduler,
                                             InsufficientFreeResources,
                                             InsufficientResourcesError)


class MyResources:
    """Simulate resources"""

    rset_dtype = [('assigned', int),  # Holds worker ID assigned to or zero
                  ('group', int)]     # Group ID this resource set belongs to

    # Basic layout
    def __init__(self, num_rsets, num_groups):
        self.total_num_rsets = num_rsets
        self.num_groups = num_groups
        self.rsets_per_node = self.total_num_rsets//num_groups
        self.even_groups = True
        self.rsets = np.zeros(self.total_num_rsets, dtype=MyResources.rset_dtype)
        self.rsets['assigned'] = 0
        for i in range(self.total_num_rsets):
            self.rsets['group'][i] = i // self.rsets_per_node
        self.rsets_free = self.total_num_rsets
        print(self.rsets)

    def free_rsets(self, worker=None):
        """Free up assigned resource sets"""
        if worker is None:
            self.rsets['assigned'] = 0
            self.rsets_free = self.total_num_rsets
        else:
            for rset, wid in enumerate(self.rsets['assigned']):
                if wid == worker:
                    self.rsets['assigned'][rset] = 0
                    self.rsets_free += 1

    def assign_rsets(self, rset_team, worker_id):
        """Mark the resource sets given by rset_team as assigned to worker_id"""
        if rset_team:
            self.rsets['assigned'][rset_team] = worker_id
            self.rsets_free -= len(rset_team)  # quick count

    # Special function for testing from a given starting point
    def fixed_assignment(self, assignment):
        """Set the given assignment along with other coupled information"""
        self.rsets['assigned'] = assignment
        self.rsets_free = np.count_nonzero(self.rsets['assigned'] == 0)


def test_too_many_rsets():
    """Tests request of more resource sets than exist"""
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    resources = MyResources(8, 2)
    sched = ResourceScheduler(user_resources=resources)

    with pytest.raises(InsufficientResourcesError):
        rset_team = sched.assign_resources(rsets_req=10)  # noqa F841
        pytest.fail('Expected InsufficientResourcesError')


def test_cannot_split_quick_return():
    """Tests the quick return when splitting finds no free even gaps"""
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    resources = MyResources(6, 3)
    resources.fixed_assignment(([1, 0, 0, 0, 3, 3]))
    sched = ResourceScheduler(user_resources=resources)

    with pytest.raises(InsufficientFreeResources):
        rset_team = sched.assign_resources(rsets_req=3)
        print(rset_team)
        pytest.fail('Expected InsufficientFreeResources')


def test_schdule_find_gaps_1node():
    """Tests assignment of rsets on one node.

    This test also checks the list is correctly assigned to workers
    and the freeing of assigned resources.
    """
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    resources = MyResources(8, 1)
    sched = ResourceScheduler(user_resources=resources)

    rset_team = sched.assign_resources(rsets_req=2)
    assert rset_team == [0, 1]

    rset_team = sched.assign_resources(rsets_req=3)
    assert rset_team == [2, 3, 4]

    # Check not enough slots
    with pytest.raises(InsufficientFreeResources):
        rset_team = sched.assign_resources(rsets_req=4)
        pytest.fail('Expected InsufficientFreeResources')

    rset_team = sched.assign_resources(rsets_req=2)
    assert rset_team == [5, 6]

    # Simulate resources freed up on return from worker
    resources.fixed_assignment(([3, 3, 0, 0, 0, 4, 4, 0]))

    # Create new scheduler to simulate new alloc call
    del sched
    sched = ResourceScheduler(user_resources=resources)

    rset_team = sched.assign_resources(rsets_req=4)
    assert rset_team == [2, 3, 4, 7]
    del resources


def test_schdule_find_gaps_2nodes():
    """Tests finding gaps on two nodes with equal resource sets"""
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    resources = MyResources(8, 2)
    sched = ResourceScheduler(user_resources=resources)
    inputs = [2, 3, 1, 2]
    exp_out = [[0, 1], [4, 5, 6], [7], [2, 3]]
    for i in range(4):
        rset_team = sched.assign_resources(rsets_req=inputs[i])
        assert rset_team == exp_out[i], \
            'Expected {}, Received rset_team {}'.format(exp_out[i], rset_team)

    with pytest.raises(InsufficientFreeResources):
        rset_team = sched.assign_resources(rsets_req=1)
        pytest.fail('Expected InsufficientFreeResources')

    del resources


def test_across_nodes_even_split():
    """Tests assignment over two nodes"""
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    resources = MyResources(8, 2)
    sched = ResourceScheduler(user_resources=resources)
    rset_team = sched.assign_resources(rsets_req=6)
    # Expecting even split
    assert rset_team == [0, 1, 2, 4, 5, 6], \
        'Even split test did not get expected result {}'.format(rset_team)

    # This time it must use 3 nodes for even split (though 2 would cover uneven).
    resources = MyResources(15, 3)
    sched = ResourceScheduler(user_resources=resources)
    rset_team = sched.assign_resources(rsets_req=9)
    # Expecting even split
    assert rset_team == [0, 1, 2, 5, 6, 7, 10, 11, 12], \
        'Even split test did not get expected result {}'.format(rset_team)
    del resources


def test_across_nodes_roundup_option():
    """Tests assignment over two nodes"""
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    resources = MyResources(8, 2)
    sched = ResourceScheduler(user_resources=resources)
    rset_team = sched.assign_resources(rsets_req=5)
    # Expecting even split
    assert rset_team == [0, 1, 2, 4, 5, 6], \
        'Even split test did not get expected result {}'.format(rset_team)
    del resources


def test_try1node_findon_2nodes():
    """Tests finding gaps on two nodes as cannot fit on one due to others assigned"""
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    resources = MyResources(8, 2)
    resources.fixed_assignment(([1, 1, 0, 0, 0, 2, 2, 0]))
    sched = ResourceScheduler(user_resources=resources)
    rset_team = sched.assign_resources(rsets_req=4)
    assert rset_team == [2, 3, 4, 7], 'rsets found {}'.format(rset_team)
    del resources


def test_try1node_findon_3nodes():
    """Tests finding gaps on two nodes as cannot fit on one due to others assigned"""
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    resources = MyResources(12, 3)
    resources.fixed_assignment(([1, 1, 0, 0, 0, 2, 2, 0, 3, 0, 3, 3]))
    sched = ResourceScheduler(user_resources=resources)

    rset_team = sched.assign_resources(rsets_req=3)
    assert rset_team == [2, 4, 9], 'rsets found {}'.format(rset_team)

    # Simulate a new call to allocation function
    sched_options = {'split2fit': False}
    del sched; sched = ResourceScheduler(user_resources=resources, sched_opts=sched_options)  # noqa E702

    with pytest.raises(InsufficientFreeResources):
        rset_team = sched.assign_resources(rsets_req=3)
        pytest.fail('Expected InsufficientFreeResources')

    # Now free up resources on 1st group and call alloc again (new sched).
    resources.free_rsets(worker=1)
    del sched; sched = ResourceScheduler(user_resources=resources, sched_opts=sched_options)  # noqa E702
    rset_team = sched.assign_resources(rsets_req=3)
    assert rset_team == [0, 1, 2], 'rsets found {}'.format(rset_team)
    del resources


def test_try2nodes_findon_3nodes():
    """Tests finding gaps on two nodes as cannot fit on one due to others assigned"""
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    resources = MyResources(18, 3)
    resources.fixed_assignment(([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]))
    sched = ResourceScheduler(user_resources=resources)

    # Can't find 2 groups of 6 so find 3 groups of 4.
    rset_team = sched.assign_resources(rsets_req=12)
    assert rset_team == [0, 2, 3, 4, 6, 7, 8, 9, 12, 13, 14, 15], 'rsets found {}'.format(rset_team)

    # Simulate a new call to allocation function
    sched_options = {'split2fit': False}
    del sched; sched = ResourceScheduler(user_resources=resources, sched_opts=sched_options)  # noqa E702
    with pytest.raises(InsufficientFreeResources):
        rset_team = sched.assign_resources(rsets_req=12)
        pytest.fail('Expected InsufficientFreeResources')

    # Now free up resources on 3rd group and call alloc again (new sched).
    resources.free_rsets(worker=3)
    del sched; sched = ResourceScheduler(user_resources=resources, sched_opts=sched_options)  # noqa E702
    rset_team = sched.assign_resources(rsets_req=12)
    assert rset_team == [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], 'rsets found {}'.format(rset_team)
    del resources


def test_split2fit_even_required_fails():
    """Tests tries one node then two, and both fail"""
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    resources = MyResources(8, 2)
    resources.fixed_assignment(([1, 1, 1, 0, 2, 2, 0, 0]))
    sched = ResourceScheduler(user_resources=resources)
    with pytest.raises(InsufficientFreeResources):
        rset_team = sched.assign_resources(rsets_req=4)  # noqa F841
        pytest.fail('Expected InsufficientFreeResources')


def test_split2fit_even_required_various():
    """Tests trying to fit to an non-even partition, and setting of local rsets_free"""
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    resources = MyResources(8, 2)
    resources.fixed_assignment(([1, 1, 1, 0, 0, 0, 0, 0]))
    sched = ResourceScheduler(user_resources=resources)
    assert sched.rsets_free == 5

    rset_team = sched.assign_resources(rsets_req=2)
    assert rset_team == [4, 5], 'rsets found {}'.format(rset_team)
    assert sched.rsets_free == 3

    # In same alloc - now try getting 4 rsets
    with pytest.raises(InsufficientFreeResources):
        rset_team = sched.assign_resources(rsets_req=4)
        pytest.fail('Expected InsufficientFreeResources')
    assert sched.rsets_free == 3

    with pytest.raises(InsufficientFreeResources):
        rset_team = sched.assign_resources(rsets_req=3)
        pytest.fail('Expected InsufficientFreeResources')
    assert sched.rsets_free == 3

    rset_team = sched.assign_resources(rsets_req=2)
    assert rset_team == [6, 7], 'rsets found {}'.format(rset_team)
    assert sched.rsets_free == 1


if __name__ == "__main__":
    test_too_many_rsets()
    test_cannot_split_quick_return()
    test_schdule_find_gaps_1node()
    test_schdule_find_gaps_2nodes()
    test_across_nodes_even_split()
    test_across_nodes_roundup_option()
    test_try1node_findon_2nodes()
    test_try1node_findon_3nodes()
    test_try2nodes_findon_3nodes()
    test_split2fit_even_required_fails()
    test_split2fit_even_required_various()
