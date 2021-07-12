import sys
import numpy as np
from libensemble.resources.scheduler import ResourceScheduler


class MyResources:
    """Simulate resources"""

    rset_dtype = [('assigned', int),  # Holds worker ID assigned to or zero
                  ('group', int)]     # Group ID this resource set belongs to

    # Basic layout
    def __init__(self, num_rsets, num_groups, ):
        self.num_rsets = num_rsets
        self.num_groups = num_groups
        self.rsets_per_node = num_rsets//num_groups
        self.even_groups = True
        self.rsets = np.zeros(self.num_rsets, dtype=MyResources.rset_dtype)
        self.rsets['assigned'] = 0
        for i in range(num_rsets):
            self.rsets['group'][i] = i // self.rsets_per_node
        self.rsets_free = self.num_rsets
        print(self.rsets)

    def free_rsets(self, worker=None):
        """Free up assigned resource sets"""
        if worker is None:
            self.rsets['assigned'] = 0
            self.rsets_free = self.num_rsets
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
    rset_team = sched.assign_resources(rsets_req=4)
    assert rset_team is None

    rset_team = sched.assign_resources(rsets_req=2)
    assert rset_team == [5, 6]

    # Simulate resources freed up on return from worker
    resources.rsets['assigned'] = [3, 3, 0, 0, 0, 4, 4, 0]

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
    rset_team = sched.assign_resources(rsets_req=1)
    assert rset_team is None
    del resources


def test_across_nodes_even_split():
    """Tests assignment over two nodes"""
    # SH TODO: This will depend on some scheduling option whether to
    #          find equal split if possible or to fill nodes first if possible.
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
    # SH TODO: This will depend on some scheduling option whether to
    #         roundup for an equal split if possible.
    print("\nTest: {}\n".format(sys._getframe().f_code.co_name))
    resources = MyResources(8, 2)
    sched = ResourceScheduler(user_resources=resources)
    rset_team = sched.assign_resources(rsets_req=5)
    # Expecting even split
    assert rset_team == [0, 1, 2, 4, 5, 6], \
        'Even split test did not get expected result {}'.format(rset_team)
    del resources

# SH TODO: Further tests for testing uneven splits - and uneven sized resource sets.


if __name__ == "__main__":
    test_schdule_find_gaps_1node()
    test_schdule_find_gaps_2nodes()
    test_across_nodes_even_split()
    test_across_nodes_roundup_option()
