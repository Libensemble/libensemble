import os
import sys
import time

import numpy as np
import pytest

from libensemble.resources.resources import Resources
from libensemble.resources.scheduler import InsufficientFreeResources, InsufficientResourcesError, ResourceScheduler


class MyResources:
    """Simulate resources"""

    rset_dtype = [
        ("assigned", int),  # Holds worker ID assigned to or zero
        ("group", int),  # Group ID this resource set belongs to
        ("slot", int),  # Slot ID this resource set belongs to
        ("gpus", bool),  # Does this resource set have GPUs
    ]

    # Basic layout
    def __init__(self, num_rsets, num_groups, gpus_per_node=0):
        self.total_num_rsets = num_rsets
        self.num_groups = num_groups
        self.rsets_per_node = self.total_num_rsets // num_groups
        self.gpu_rsets_per_node = min(gpus_per_node, self.rsets_per_node)
        self.nongpu_rsets_per_node = self.rsets_per_node - self.gpu_rsets_per_node

        self.even_groups = True
        self.rsets = np.zeros(self.total_num_rsets, dtype=MyResources.rset_dtype)
        self.rsets["assigned"] = -1
        for i in range(self.total_num_rsets):
            self.rsets["group"][i] = i // self.rsets_per_node
            self.rsets["slot"][i] = i % self.rsets_per_node
            self.rsets["gpus"][i] = self.rsets["slot"][i] < gpus_per_node
        self.rsets_free = self.total_num_rsets
        self.total_num_gpu_rsets = np.count_nonzero(self.rsets["gpus"])
        self.total_num_nongpu_rsets = np.count_nonzero(~self.rsets["gpus"])

        self.gpu_rsets_free = self.total_num_gpu_rsets
        self.nongpu_rsets_free = self.total_num_nongpu_rsets

    def free_rsets(self, worker=None):
        """Free up assigned resource sets"""
        if worker is None:
            self.rsets["assigned"] = -1
            self.rsets_free = self.total_num_rsets
        else:
            for rset, wid in enumerate(self.rsets["assigned"]):
                if wid == worker:
                    self.rsets["assigned"][rset] = -1
                    self.rsets_free += 1

    def assign_rsets(self, rset_team, worker_id):
        """Mark the resource sets given by rset_team as assigned to worker_id"""
        if rset_team:
            self.rsets["assigned"][rset_team] = worker_id
            self.rsets_free -= len(rset_team)  # quick count

    # Special function for testing from a given starting point
    def fixed_assignment(self, assignment):
        """Set the given assignment along with other coupled information"""
        self.rsets["assigned"] = assignment
        self.rsets_free = np.count_nonzero(self.rsets["assigned"] == -1)


def _fail_to_resource(sched, rsets, use_gpus=None):
    with pytest.raises(InsufficientFreeResources):
        rset_team = sched.assign_resources(rsets_req=rsets, use_gpus=use_gpus)
        pytest.fail(f"Expected InsufficientFreeResources. Found {rset_team}")


def _print_assigned(resources):
    """For debugging. Print assigned rsets by group"""
    rsets = resources.rsets
    max_groups = max(rsets["group"])
    print("\nAssigned")
    for g in range(max_groups + 1):
        filt = rsets["group"] == g
        print(rsets["assigned"][filt])
    print(f"free rsets {resources.free_rsets}\n")


def test_request_zero_rsets():
    """Tests requesting zero resource sets"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    resources = MyResources(8, 2)

    # No options
    sched = ResourceScheduler(user_resources=resources)
    rset_team = sched.assign_resources(rsets_req=0)
    assert rset_team == [], f"rset_team is {rset_team}. Expected zero"
    del sched
    rset_team = None

    # Options should make no difference
    for match_slots in [False, True]:
        for split2fit in [False, True]:
            sched_options = {"match_slots": match_slots, "split2fit": split2fit}
            sched = ResourceScheduler(user_resources=resources, sched_opts=sched_options)
            rset_team = sched.assign_resources(rsets_req=0)
            assert rset_team == [], f"rset_team is {rset_team}. Expected zero"
            del sched
            rset_team = None
    del resources


def _fail_too_many_rsets(resources, use_gpus=None):
    # No options
    sched = ResourceScheduler(user_resources=resources)

    with pytest.raises(InsufficientResourcesError):
        rset_team = sched.assign_resources(rsets_req=10, use_gpus=use_gpus)  # noqa F841
        pytest.fail("Expected InsufficientResourcesError")

    del sched
    rset_team = None

    # Options should make no difference
    for match_slots in [False, True]:
        for split2fit in [False, True]:
            sched_options = {"match_slots": match_slots, "split2fit": split2fit}
            sched = ResourceScheduler(user_resources=resources, sched_opts=sched_options)
            with pytest.raises(InsufficientResourcesError):
                rset_team = sched.assign_resources(rsets_req=10, use_gpus=use_gpus)  # noqa F841
                pytest.fail("Expected InsufficientResourcesError")
            del sched


def test_too_many_rsets():
    """Tests request of more resource sets than exist"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")

    resources = MyResources(8, 2)
    _fail_too_many_rsets(resources)
    del resources

    resources = MyResources(12, 2, 4)
    _fail_too_many_rsets(resources, use_gpus=True)
    _fail_too_many_rsets(resources, use_gpus=False)
    del resources


def test_cannot_split_quick_return():
    """Tests the quick return when splitting finds no free even gaps"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    resources = MyResources(6, 3)
    resources.fixed_assignment(([1, -1, -1, -1, 3, 3]))
    sched = ResourceScheduler(user_resources=resources)
    _fail_to_resource(sched, 3)


def test_schedule_find_gaps_1node():
    """Tests assignment of rsets on one node.

    This test also checks the list is correctly assigned to workers
    and the freeing of assigned resources.
    """
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    resources = MyResources(8, 1)

    # Options should make no difference
    for match_slots in [False, True]:
        for split2fit in [False, True]:
            sched_options = {"match_slots": match_slots, "split2fit": split2fit}
            sched = ResourceScheduler(user_resources=resources, sched_opts=sched_options)

            rset_team = sched.assign_resources(rsets_req=2)
            assert rset_team == [0, 1], f"rset_team is {rset_team}"

            rset_team = sched.assign_resources(rsets_req=3)
            assert rset_team == [2, 3, 4]

            # Check not enough slots
            _fail_to_resource(sched, 4)

            rset_team = sched.assign_resources(rsets_req=2)
            assert rset_team == [5, 6]

            # Simulate resources freed up on return from worker
            resources.fixed_assignment(([3, 3, -1, -1, -1, 4, 4, -1]))

            # Create new scheduler to simulate new alloc call
            del sched
            rset_team = None
            sched = ResourceScheduler(user_resources=resources)

            rset_team = sched.assign_resources(rsets_req=4)
            assert rset_team == [2, 3, 4, 7]

            del sched
            rset_team = None
            resources.free_rsets()
    del resources


def test_schedule_find_gaps_2nodes():
    """Tests finding gaps on two nodes with equal resource sets"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    resources = MyResources(8, 2)
    inputs = [2, 3, 1, 2]
    exp_out = [[0, 1], [4, 5, 6], [7], [2, 3]]

    for match_slots in [False, True]:
        for split2fit in [False, True]:
            sched_options = {"match_slots": match_slots, "split2fit": split2fit}
            sched = ResourceScheduler(user_resources=resources, sched_opts=sched_options)
            for i in range(4):
                rset_team = sched.assign_resources(rsets_req=inputs[i])
                assert rset_team == exp_out[i], f"Expected {exp_out[i]}, Received rset_team {rset_team}"
            _fail_to_resource(sched, 1)
            del sched
            rset_team = None
    del resources


def test_split_across_no_matching_slots():
    """Must split across - but no split2fit and no matching slots"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    resources = MyResources(6, 3)  # 3 nodes of 2 slots

    for split2fit in [False, True]:
        resources.fixed_assignment(([-1, 1, 1, -1, -1, 1]))
        sched_options = {"split2fit": split2fit}
        sched = ResourceScheduler(user_resources=resources, sched_opts=sched_options)
        _fail_to_resource(sched, 3)

        sched.match_slots = False
        rset_team = sched.assign_resources(rsets_req=3)
        assert rset_team == [0, 3, 4], f"rset_team is {rset_team}."
        del sched
        rset_team = None
    del resources


def test_across_nodes_even_split():
    """Tests even assignment over two and three nodes

    Also tests cached variables in scheduler.
    """
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")

    # Options should make no difference
    for match_slots in [False, True]:
        for split2fit in [False, True]:
            resources = MyResources(8, 2)
            sched_options = {"match_slots": match_slots, "split2fit": split2fit}
            sched = ResourceScheduler(user_resources=resources, sched_opts=sched_options)

            rset_team = sched.assign_resources(rsets_req=6)
            # Expecting even split
            assert rset_team == [0, 1, 2, 4, 5, 6], f"Even split test did not get expected result {rset_team}"
            assert sched.rsets_free == 2, f"rsets_free should be 2. Found {sched.rsets_free}"
            assert sched.avail_rsets_by_group == {0: [3], 1: [7]}

            # Now find the remaining 2 slots
            if not sched_options["split2fit"]:
                _fail_to_resource(sched, 2)
            else:
                rset_team = sched.assign_resources(rsets_req=2)
                assert rset_team == [3, 7], f"rsets found {rset_team}"
                assert sched.rsets_free == 0
                assert sched.avail_rsets_by_group == {0: [], 1: []}
            del sched
            rset_team = None

            # This time it must use 3 nodes for even split (though 2 would cover uneven).
            resources = MyResources(15, 3)
            sched = ResourceScheduler(user_resources=resources, sched_opts=sched_options)
            rset_team = sched.assign_resources(rsets_req=9)
            # Expecting even split
            # Even split requirement means even if ``split2fit`` is False, will still split to 3x3
            assert rset_team == [0, 1, 2, 5, 6, 7, 10, 11, 12], "Even split test did not get expected result {}".format(
                rset_team
            )
            if not sched_options["split2fit"]:
                _fail_to_resource(sched, 6)
            else:
                rset_team = sched.assign_resources(rsets_req=6)
                assert rset_team == [3, 4, 8, 9, 13, 14], f"rsets found {rset_team}"

            del sched
            rset_team = None
    del resources


def test_across_nodes_roundup_option_2nodes():
    """Tests assignment over two nodes"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    resources = MyResources(8, 2)

    # Options should make no difference
    for match_slots in [False, True]:
        for split2fit in [False, True]:
            sched_options = {"match_slots": match_slots, "split2fit": split2fit}
            sched = ResourceScheduler(user_resources=resources, sched_opts=sched_options)
            rset_team = sched.assign_resources(rsets_req=5)
            # Expecting even split
            assert rset_team == [0, 1, 2, 4, 5, 6], f"Even split test did not get expected result {rset_team}"
            assert sched.rsets_free == 2, f"Free slots found {sched.rsets_free}"
            del sched
            rset_team = None
    del resources


def test_across_nodes_roundup_option_3nodes():
    """Tests assignment over two nodes"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    resources = MyResources(9, 3)

    # Options should make no difference
    for match_slots in [False, True]:
        for split2fit in [False, True]:
            sched_options = {"match_slots": match_slots, "split2fit": split2fit}
            sched = ResourceScheduler(user_resources=resources, sched_opts=sched_options)
            rset_team = sched.assign_resources(rsets_req=7)
            # Expecting even split
            assert rset_team == [0, 1, 2, 3, 4, 5, 6, 7, 8], "Even split test did not get expected result {}".format(
                rset_team
            )
            assert sched.rsets_free == 0, f"Free slots found {sched.rsets_free}"
            del sched
            rset_team = None
    del resources


def test_try1node_findon_2nodes_matching_slots():
    """Tests finding gaps on two nodes with matching slots"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    resources = MyResources(8, 2)

    fixed_assignments = [
        ([1, 1, -1, -1, 3, 3, -1, -1]),
        ([-1, 1, -1, 2, -1, 4, -1, 4]),
        ([-1, 1, 1, -1, -1, 2, 2, -1]),
        ([1, -1, 1, -1, 3, -1, 3, -1]),
    ]
    exp_out = [[2, 3, 6, 7], [0, 2, 4, 6], [0, 3, 4, 7], [1, 3, 5, 7]]

    for i, assgn in enumerate(fixed_assignments):
        resources.fixed_assignment(assgn)

        # match_slots should make no difference
        for match_slots in [False, True]:
            sched_options = {"match_slots": match_slots}
            sched = ResourceScheduler(user_resources=resources, sched_opts=sched_options)
            rset_team = sched.assign_resources(rsets_req=4)
            assert rset_team == exp_out[i], "Expected {}, Received rset_team {} - match_slots is {}".format(
                exp_out[i], rset_team, match_slots
            )
        del sched
        rset_team = None
    del resources


def test_try1node_findon_2nodes_different_slots():
    """Tests finding gaps on two nodes with non-matching slots"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    resources = MyResources(8, 2)

    fixed_assignments = [
        ([1, 1, -1, -1, -1, 2, 2, -1]),
        ([1, 1, -1, -1, -1, -1, 3, 3]),
        ([1, -1, -1, 1, -1, 3, -1, 3]),
    ]
    exp_out = [[2, 3, 4, 7], [2, 3, 4, 5], [1, 2, 4, 6]]

    for i, assgn in enumerate(fixed_assignments):
        resources.fixed_assignment(assgn)
        sched = ResourceScheduler(user_resources=resources)

        # Default options - cannot match slots
        _fail_to_resource(sched, 4)

        # Quick change option in scheduler directly - cannot split2fit.
        sched.match_slots = False
        sched.split2fit = False
        _fail_to_resource(sched, 4)

        # Now with match slots False and split2fit True - should find.
        sched.split2fit = True
        rset_team = sched.assign_resources(rsets_req=4)
        assert rset_team == exp_out[i], f"Expected {exp_out[i]}, Received rset_team {rset_team}"

        del sched
        rset_team = None
    del resources


def test_try1node_findon_3nodes():
    """Tests finding gaps on two nodes as cannot fit on one due to others assigned"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    resources = MyResources(12, 3)
    resources.fixed_assignment(([1, 1, -1, -1, -1, 2, 2, -1, 3, -1, 3, 3]))
    sched = ResourceScheduler(user_resources=resources)

    # Default - with match_slots - cannot find a split with matching slots
    _fail_to_resource(sched, 3)

    # Can find non-matching slots across three nodes
    sched_options = {"match_slots": False}
    del sched
    sched = ResourceScheduler(user_resources=resources, sched_opts=sched_options)
    rset_team = sched.assign_resources(rsets_req=3)
    assert rset_team == [2, 4, 9], f"rsets found {rset_team}"

    # Without split2fit, will not split over nodes.
    sched_options = {"match_slots": False, "split2fit": False}
    del sched
    sched = ResourceScheduler(user_resources=resources, sched_opts=sched_options)
    _fail_to_resource(sched, 3)

    # Now free up resources on 1st group and call alloc again (new sched as change to resources).
    resources.free_rsets(worker=1)
    del sched
    sched = ResourceScheduler(user_resources=resources, sched_opts=sched_options)
    rset_team = sched.assign_resources(rsets_req=3)
    assert rset_team == [0, 1, 2], f"rsets found {rset_team}"
    del resources


def test_try2nodes_findon_3nodes():
    """Tests finding gaps on two nodes as cannot fit on one due to others assigned

    Assigned means assigned to workerID (1 and 3 in this case).

    12 resource sets requires 2 nodes. But if some are already assigned, and if
    ``split2fit`` is True (default), then can split to 3 nodes, which finds an
    even split (3 nodes of 4). If ``match_slots`` is True (default),
    then must find matching slots on each node. Otherwise, a simple algorithm will
    find the first available slots on each node.

    match slots:    x           x      x           x       x           x
    rset ID:     0  1  2  3  4  5   6  7  8  9  10 11   12 13 14 15 16 17
    slots ID:    0  1  2  3  4  5   0  1  2  3  4  5    0  1  2  3  4  5
    assigned:    0  1  0  0  0  0   0  0  0  0  0  0    0  0  0  0  0  3

    After releasing worker 3, there is no need to split to three nodes.

    match slots:    x
    rset ID:     0  1  2  3  4  5   6  7  8  9  10 11   12 13 14 15 16 17
    slots ID:    0  1  2  3  4  5   0  1  2  3  4  5    0  1  2  3  4  5
    assigned:    0  1  0  0  0  0   0  0  0  0  0  0    0  0  0  0  0  0

    """
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    resources = MyResources(18, 3)
    resources.fixed_assignment(([-1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 3]))
    sched = ResourceScheduler(user_resources=resources)

    # Can't find 2 groups of 6 so find 3 groups of 4 - with matching slots.
    rset_team = sched.assign_resources(rsets_req=12)
    assert rset_team == [0, 2, 3, 4, 6, 8, 9, 10, 12, 14, 15, 16], f"rsets found {rset_team}"

    # Without matching slots, will find first available slots on each node.
    sched_options = {"match_slots": False}
    del sched
    sched = ResourceScheduler(user_resources=resources, sched_opts=sched_options)
    rset_team = sched.assign_resources(rsets_req=12)
    assert rset_team == [0, 2, 3, 4, 6, 7, 8, 9, 12, 13, 14, 15], f"rsets found {rset_team}"

    # Simulate a new call to allocation function with split2fit False - unable to split to 3 nodes.
    sched_options = {"match_slots": False, "split2fit": False}
    del sched
    sched = ResourceScheduler(user_resources=resources, sched_opts=sched_options)
    _fail_to_resource(sched, 12)

    # Now free up resources on 3rd group and call alloc again (new sched).
    # Even without split2fit - will split to two nodes (as 12 requires two nodes even when empty).
    resources.free_rsets(worker=3)
    for match_slots in [False, True]:
        for split2fit in [False, True]:
            sched_options = {"match_slots": match_slots, "split2fit": split2fit}

            del sched
            sched = ResourceScheduler(user_resources=resources, sched_opts=sched_options)
            rset_team = sched.assign_resources(rsets_req=12)
            assert rset_team == [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], f"rsets found {rset_team}"

            del sched
            sched = ResourceScheduler(user_resources=resources, sched_opts=sched_options)
            rset_team = sched.assign_resources(rsets_req=12)
            assert rset_team == [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], f"rsets found {rset_team}"
    del resources


def test_split2fit_even_required_fails():
    """Test tries one node then two, and both fail"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    resources = MyResources(8, 2)
    resources.fixed_assignment(([1, 1, 1, -1, 2, 2, -1, -1]))

    for match_slots in [False, True]:
        sched_options = {"match_slots": match_slots}
        sched = ResourceScheduler(user_resources=resources, sched_opts=sched_options)
        _fail_to_resource(sched, 4)
        assert sched.rsets_free == 3


def test_split2fit_even_required_various():
    """Tests trying to fit to an non-even partition, and setting of local rsets_free"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    resources = MyResources(8, 2)
    resources.fixed_assignment(([1, 1, 1, -1, -1, -1, -1, -1]))
    sched = ResourceScheduler(user_resources=resources)
    assert sched.rsets_free == 5

    rset_team = sched.assign_resources(rsets_req=2)
    assert rset_team == [4, 5], f"rsets found {rset_team}"
    assert sched.rsets_free == 3

    # In same alloc - now try getting 4 rsets, then 3
    _fail_to_resource(sched, 4)
    assert sched.rsets_free == 3
    _fail_to_resource(sched, 3)
    assert sched.rsets_free == 3

    rset_team = sched.assign_resources(rsets_req=2)
    assert rset_team == [6, 7], f"rsets found {rset_team}"
    assert sched.rsets_free == 1


def test_try1node_findon_2_or_4nodes():
    """Tests splitting to fit. Needs 4 nodes if matching slots, else 2."""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    resources = MyResources(16, 4)
    resources.fixed_assignment(([1, 1, -1, 1, 2, 2, -1, -1, 1, -1, -1, 1, -1, 4, -1, 4]))

    sched = ResourceScheduler(user_resources=resources)
    rset_team = sched.assign_resources(rsets_req=4)
    assert rset_team == [2, 6, 10, 14], f"rsets found {rset_team}"
    del sched
    rset_team = None  # I think should always do between tests (esp if expected output is the same).

    sched_options = {"match_slots": False}  # will prob be default.
    sched = ResourceScheduler(user_resources=resources, sched_opts=sched_options)  # noqa E702
    rset_team = sched.assign_resources(rsets_req=4)
    assert rset_team == [6, 7, 9, 10], f"rsets found {rset_team}"
    del resources


def _construct_large_problem(resources):
    """Constructs rset assignment for large problem"""
    rsets = resources.rsets

    # All slots filled
    rsets["assigned"] = 1

    # Now free up the one column
    col15 = rsets["slot"] == 15
    rsets["assigned"][col15] = -1

    # Now make sure two rows (groups) with 8, but different slots
    free_row0 = (rsets["group"] == 0) & (rsets["slot"] < 8)
    free_row1 = (rsets["group"] == 1) & (rsets["slot"] >= 8)

    # Now make sure 4 rows of 4 exist (diff slots)
    free_row2 = (rsets["group"] == 2) & (rsets["slot"] < 4)
    free_row3 = (rsets["group"] == 3) & (rsets["slot"] >= 8) & (rsets["slot"] < 12)

    # Now make sure 8 rows of 2 exist (diff slots)
    # Free one slot each as last column already free
    free_strip = (rsets["group"] >= 12) & (rsets["slot"] == 3)

    rsets["assigned"][free_row0] = -1
    rsets["assigned"][free_row1] = -1
    rsets["assigned"][free_row2] = -1
    rsets["assigned"][free_row3] = -1
    rsets["assigned"][free_strip] = -1

    resources.free_rsets = np.count_nonzero(rsets["assigned"] == -1)
    # _print_assigned(resources)


def test_large_match_slots():
    """Tests multiple match slots iterations

    Aim is try one of 16, then 2 of 8, then 4 or 4 and 8 or 2 then 16 of one.
    To do this need enough slots at each step so tries to find, but in the
    wrong places, until final iteration. Performance is of interest.
    """
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")

    # Construct rset assignment
    resources = MyResources(256, 16)
    _construct_large_problem(resources)

    exp_out = [
        [0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31],
        [15, 31, 47, 63, 79, 95, 111, 127, 143, 159, 175, 191, 207, 223, 239, 255],
    ]

    for match_slots in [False, True]:
        sched_options = {"match_slots": match_slots}
        sched = ResourceScheduler(user_resources=resources, sched_opts=sched_options)
        time1 = time.time()
        rset_team = sched.assign_resources(rsets_req=16)
        time2 = time.time() - time1
        assert rset_team == exp_out[match_slots], "Expected {}, Received rset_team {}".format(
            exp_out[match_slots], rset_team
        )
        print(f"Time for large problem (match_slots {match_slots}): {time2}")
        del sched
        rset_team = None
    del resources


def test_schedule_find_gaps_2nodes_withgpus():
    """Tests finding gaps on two nodes with equal resource sets"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    resources = MyResources(8, 2, 3)  # 3 gpu rsets per node.

    inputs = [2, 2, 2]
    exp_out = [[0, 1], [4, 5], [2, 6]]
    exp_rsets_free = [6, 4, 2]
    exp_gpu_rsets_free = [4, 2, 0]
    exp_nongpu_rsets_free = [2, 2, 2]

    num_trials = len(inputs)

    for match_slots in [False, True]:
        for split2fit in [False, True]:
            sched_options = {"match_slots": match_slots, "split2fit": split2fit}
            sched = ResourceScheduler(user_resources=resources, sched_opts=sched_options)

            for i in range(num_trials):
                if not sched_options["split2fit"] and i == 2:
                    expect_failure = True
                else:
                    expect_failure = False

                if expect_failure:
                    _fail_to_resource(sched, 2, True)
                    exp_i = i - 1  # free resource sets should be same as previously
                else:
                    rset_team = sched.assign_resources(rsets_req=inputs[i], use_gpus=True)
                    assert rset_team == exp_out[i], f"Expected {exp_out[i]}, Received rset_team {rset_team}"
                    exp_i = i

                assert sched.rsets_free == exp_rsets_free[exp_i], f"rsets_free was {sched.rsets_free}"
                assert sched.gpu_rsets_free == exp_gpu_rsets_free[exp_i], f"gpu_rsets_free was {sched.gpu_rsets_free}"
                assert (
                    sched.nongpu_rsets_free == exp_nongpu_rsets_free[exp_i]
                ), f"nongpu_rsets_free was {sched.nongpu_rsets_free}"

            if expect_failure:
                # then here expect success
                rset_team = sched.assign_resources(rsets_req=1, use_gpus=True)
                assert rset_team == [2], f"Expected 2, Received rset_team {rset_team}"
            else:
                _fail_to_resource(sched, 1, True)
                rset_team = sched.assign_resources(rsets_req=2, use_gpus=False)
                assert rset_team == [3, 7], f"Expected [3, 7], Received rset_team {rset_team}"

                # All rsets used
                assert sched.rsets_free == 0, f"rsets_free was {sched.rsets_free}"
                assert sched.gpu_rsets_free == 0, f"gpu_rsets_free was {sched.gpu_rsets_free}"
                assert sched.nongpu_rsets_free == 0, f"nongpu_rsets_free was {sched.nongpu_rsets_free}"

            del sched
            rset_team = None
    del resources


def _setup_resources_sharing(gpu_sharing_factor=2, gpus_per_group=None, sched_opts=None, num_resource_sets=16):
    """Create a mocked 2-node, 4-GPU/node resource set for helper tests.

    Mocks SLURM nodelist ``knl-[0020,0036]`` (2 nodes) with
    ``gpus_on_node=4`` and ``cores_on_node=(32,64)``-equivalent layout via
    ``ResourceInfo``. With ``gpu_sharing_factor=2`` each physical GPU is
    split into 2 logical shares, yielding ``8`` GPU shares per node and
    ``16`` total for the default ``num_resource_sets=16`` (``8`` per node).
    Returns the ``ResourceManager`` and a ``ResourceScheduler`` bound to it.

    Args:
        gpu_sharing_factor: Logical shares per physical GPU (default 2).
        gpus_per_group: Optional cap on GPUs per scheduler group, scaled by
            ``gpu_sharing_factor`` in ``RSetResources.get_group_list``.
        sched_opts: Dict of scheduler options, notably ``gpu_packing``
            ``\"pack\"`` (co-locate fractional shares) or ``\"spread\"``.
        num_resource_sets: Total rsets to create; ``rsets_per_node`` is
            derived as ``ceil(num_resource_sets/num_nodes)``.

    Returns:
        Tuple of (resource_manager, scheduler) ready for direct calls to
        ``_build_gpu_bins`` etc. Caller is responsible for cleaning the
        mocked env var if needed.
    """
    # Mock a 2-node SLURM nodelist; no real hardware needed
    os.environ["LIBE_RESOURCES_TEST_NODE_LIST"] = "knl-[0020,0036]"
    resource_info = {"nodelist_env_slurm": "LIBE_RESOURCES_TEST_NODE_LIST", "gpus_on_node": 4}
    libE_specs = {
        "resource_info": resource_info,
        "dedicated_mode": False,  # keep libEnsemble ranks out of the nodelist
        "gpu_sharing_factor": gpu_sharing_factor,
        "gpus_per_group": gpus_per_group,
        "num_resource_sets": num_resource_sets,
    }
    # Instantiate Resources and its ResourceManager (holds RSetResources)
    res = Resources(libE_specs=libE_specs, top_level_dir=os.getcwd())
    res.set_resource_manager(num_workers=8)
    # Bind scheduler to the mocked manager
    sched = ResourceScheduler(user_resources=res.resource_manager, sched_opts=sched_opts or {})
    return res.resource_manager, sched


def test_build_gpu_bins():
    """Unit coverage for ``ResourceScheduler._build_gpu_bins``.

    Verifies the per-group GPU bin construction on a fresh 2-node, 16-rset
    layout (sf=2). Each physical GPU should own exactly ``sf`` rsets, so
    group 1 is expected to contain bins ``{0:[0,1], 1:[2,3], 2:[4,5],
    3:[6,7]}`` and group 2 similarly ``11:[10,11]`` … ``14,15``. This
    structure is the foundation for all physical-exclusive / packing
    decisions.
    """
    # Fresh mocked resources: 2 groups, 8 GPU shares each
    _, sched = _setup_resources_sharing()
    # Get GPU-only availability by group
    valid = sched.get_avail_rsets_by_group()
    valid = sched.filter_for_rset_type(valid, use_gpus=True)
    # Build bins: group -> gpu_id -> [rset_ids]
    bins = sched._build_gpu_bins(valid)
    # Two groups (nodes) expected
    assert set(bins.keys()) == {1, 2}
    # Group 1: 4 physical GPUs each with 2 shares
    assert bins[1][0] == [0, 1]  # GPU 0 shares
    assert bins[1][1] == [2, 3]  # GPU 1 shares
    assert bins[1][2] == [4, 5]  # GPU 2 shares
    assert bins[1][3] == [6, 7]  # GPU 3 shares
    # Spot-check group 2 last GPU
    assert bins[2][3] == [14, 15]


def test_can_group_satisfy_gpu():
    """Unit coverage for ``ResourceScheduler._can_group_satisfy_gpu``.

    Exercises every branch of the satisfiability check:

    * **Fresh node:** ``need`` 1,2,4,8 (1 share … all 8 shares on one node)
      should succeed, 9 should fail.
    * **After partial allocation:** consume 2 shares (one whole GPU), then
      6 shares still fits but 7 does not.
    * **After one-share allocation:** 1 share still fits, 2 still needs a
      fully-free GPU.
    * **Synthetic bins:** a partially-occupied GPU ``{0:[0]}`` with others
      fully free still satisfies 1 and 3 shares via the ``leftover_fully``
      and ``partial >= rem`` paths.
    * **Fragmented failure:** bins each with a single free share cannot
      satisfy a whole GPU (need 2) or 5 shares, hitting the ``total_free``
      fallback.
    """
    # Start from a fully-free 8-share-per-node layout
    _, sched = _setup_resources_sharing()
    valid = sched.filter_for_rset_type(sched.get_avail_rsets_by_group(), use_gpus=True)
    bins = sched._build_gpu_bins(valid)
    # Fresh node satisfiability: 1 share up to full node, 9 exceeds capacity
    assert sched._can_group_satisfy_gpu(1, 1, bins) is True  # single share
    assert sched._can_group_satisfy_gpu(1, 2, bins) is True  # one whole GPU (sf=2)
    assert sched._can_group_satisfy_gpu(1, 4, bins) is True  # two whole GPUs
    assert sched._can_group_satisfy_gpu(1, 8, bins) is True  # entire node
    assert sched._can_group_satisfy_gpu(1, 9, bins) is False  # over capacity
    # Allocate one whole GPU (2 shares) and re-check remaining capacity
    sched.assign_resources(rsets_req=2, use_gpus=True)
    valid2 = sched.filter_for_rset_type(sched.get_avail_rsets_by_group(), use_gpus=True)
    bins2 = sched._build_gpu_bins(valid2)
    assert sched._can_group_satisfy_gpu(1, 6, bins2) is True  # 6 of remaining 6 fits
    assert sched._can_group_satisfy_gpu(1, 7, bins2) is False  # 7 needs more than free
    # One share taken from fresh manager: still room for 1 or 2 on same node
    sched2 = ResourceScheduler(user_resources=_setup_resources_sharing()[0], sched_opts={"gpu_packing": "pack"})
    sched2.assign_resources(rsets_req=1, use_gpus=True)
    valid3 = sched2.filter_for_rset_type(sched2.get_avail_rsets_by_group(), use_gpus=True)
    bins3 = sched2._build_gpu_bins(valid3)
    assert sched2._can_group_satisfy_gpu(1, 1, bins3) is True
    assert sched2._can_group_satisfy_gpu(1, 2, bins3) is True
    # Synthetic bins with one partially-occupied GPU: triggers leftover_fully branch
    bins_partial = {1: {0: [0], 1: [2, 3], 2: [4, 5], 3: [6, 7]}}
    assert sched._can_group_satisfy_gpu(1, 1, bins_partial) is True  # uses leftover fully-free GPU
    assert sched._can_group_satisfy_gpu(1, 3, bins_partial) is True  # needs 1 whole +1, partial suffices
    # Highly fragmented: each GPU has only 1 free share, no whole GPU left
    bins_frag = {1: {0: [0], 1: [2], 2: [4], 3: [6]}}
    assert sched._can_group_satisfy_gpu(1, 2, bins_frag) is False  # need whole GPU
    assert sched._can_group_satisfy_gpu(1, 5, bins_frag) is False  # total_free insufficient


def test_select_rsets_for_gpu_group_pack_spread():
    """Unit coverage for ``ResourceScheduler._select_rsets_for_gpu_group``.

    Checks both ``pack`` and ``spread`` policies inside a single group:

    * **Fresh group:** 1 share picks ``[0]``; 2 shares (one whole GPU)
      picks two rsets on the same ``gpu_id``.
    * **Pack after one taken:** with ``[0]`` consumed, pack co-locates the
      next 1-share request onto the same GPU ``[1]`` to preserve fully-free
      GPUs.
    * **Spread after one taken:** spread instead picks the least-occupied
      GPU ``[2]`` (next fully-free GPU).
    * **Whole + remainder:** 4 shares picks 2 whole GPUs (distinct ids), 3
      shares picks whole+1 (also 2 distinct ids). These validate the
      ``whole``/``rem`` split and the candidate sorting.
    """
    # Pack policy fresh group
    _, sched_pack = _setup_resources_sharing(sched_opts={"gpu_packing": "pack"})
    valid = sched_pack.filter_for_rset_type(sched_pack.get_avail_rsets_by_group(), use_gpus=True)
    bins = sched_pack._build_gpu_bins(valid)
    # 1 share on fresh node: picks first rset of first GPU (lowest gid)
    sel = sched_pack._select_rsets_for_gpu_group(1, 1, bins)
    assert sel == [0]
    # 2 shares = one whole GPU: both rsets on same physical GPU
    sel2 = sched_pack._select_rsets_for_gpu_group(1, 2, bins)
    assert len(sel2) == 2
    assert sched_pack.resources.rsets[sel2[0]]["gpu_id"] == sched_pack.resources.rsets[sel2[1]]["gpu_id"]
    # Pack co-location: after [0] taken, next 1-share should fill same GPU [1]
    sched_pack.assign_resources(rsets_req=1, use_gpus=True)
    valid2 = sched_pack.filter_for_rset_type(sched_pack.get_avail_rsets_by_group(), use_gpus=True)
    bins2 = sched_pack._build_gpu_bins(valid2)
    sel_pack = sched_pack._select_rsets_for_gpu_group(1, 1, bins2)
    assert sel_pack == [1]  # preserves fully-free GPUs
    # Spread: same scenario should pick next fully-free GPU instead
    _, sched_spread = _setup_resources_sharing(sched_opts={"gpu_packing": "spread"})
    sched_spread.assign_resources(rsets_req=1, use_gpus=True)
    valid_s = sched_spread.filter_for_rset_type(sched_spread.get_avail_rsets_by_group(), use_gpus=True)
    bins_s = sched_spread._build_gpu_bins(valid_s)
    sel_spread = sched_spread._select_rsets_for_gpu_group(1, 1, bins_s)
    assert sel_spread == [2]  # least-occupied fully-free GPU
    # Whole GPUs: 4 shares => 2 distinct physical GPUs
    sel_whole = sched_pack._select_rsets_for_gpu_group(1, 4, bins)
    assert len(sel_whole) == 4
    assert len(set(sched_pack.resources.rsets[sel_whole]["gpu_id"])) == 2
    # Whole+remainder: 3 shares => 1 whole GPU +1 share on another GPU
    sel_rem = sched_pack._select_rsets_for_gpu_group(1, 3, bins)
    assert len(sel_rem) == 3
    assert len(set(sched_pack.resources.rsets[sel_rem]["gpu_id"])) == 2


def test_select_rsets_fragmented_remainder():
    """Covers fragmented-remainder path in ``_select_rsets_for_gpu_group``.

    This is the ``else`` branch at ``scheduler.py:256-269`` which was
    previously uncovered. It triggers when only *partial* GPUs are available
    — no single physical GPU has enough free shares to satisfy ``rem`` — so
    the remainder must be assembled from fragments across multiple GPUs.

    * **Only partial GPUs:** with ``sf=4``, ``need=3`` (``whole=0, rem=3``) and
      bins of size ``1,2,1`` no candidate satisfies ``len>=rem``; the method
      fragments ``rem`` across GPUs. ``pack`` picks smallest bins first,
      ``spread`` picks largest first — both are verified.
    * **Whole + fragmented rem:** ``need=7`` (``whole=1, rem=3``) consumes one
      fully-free GPU (4 shares) then fragments the remaining 3 as above.
    * **Error:** fragmented capacity insufficient (need 3 but only 2 free)
      correctly raises ``InsufficientFreeResources``.
    """
    # Use sf=4 so rem can be 3 and fragmentation is observable (sf=2 never fragments)
    res4, _ = _setup_resources_sharing(gpu_sharing_factor=4, num_resource_sets=32)
    sched_pack = ResourceScheduler(user_resources=res4, sched_opts={"gpu_packing": "pack"})
    sched_spread = ResourceScheduler(user_resources=res4, sched_opts={"gpu_packing": "spread"})

    # Fragmented remainder only: need=3, whole=0, rem=3, max bin size 2 <3 → no candidates
    bins_frag = {1: {0: [0], 1: [4, 5], 2: [8]}}  # lens 1,2,1
    sel_pack = sched_pack._select_rsets_for_gpu_group(1, 3, bins_frag)
    # pack sorts by (len, gid): (1,0)->[0], (1,2)->[8], (2,1)->[4] => [0,8,4] sorted
    assert sel_pack == [0, 4, 8]
    sel_spread = sched_spread._select_rsets_for_gpu_group(1, 3, bins_frag)
    # spread sorts by (-len, gid): (2,1)->[4,5] (take 2), (1,0)->[0] (take 1) => [4,5,0] sorted
    assert sel_spread == [0, 4, 5]

    # Whole + fragmented remainder: need=7 → whole=1 (4 shares), rem=3 fragmented
    bins_whole_frag = {1: {0: [0, 1, 2, 3], 1: [4], 2: [8], 3: [12, 13]}}
    sel_pack2 = sched_pack._select_rsets_for_gpu_group(1, 7, bins_whole_frag)
    # whole picks gid 0 (fully free): [0,1,2,3], remainder fragments 1,1,2 → [4,8,12] (pack order)
    assert sel_pack2 == [0, 1, 2, 3, 4, 8, 12]
    sel_spread2 = sched_spread._select_rsets_for_gpu_group(1, 7, bins_whole_frag)
    # spread picks largest remainder first: gid3 (2) then gid1 (1)
    assert sel_spread2 == [0, 1, 2, 3, 4, 12, 13]

    # Error path: not enough fragmented capacity for rem
    bins_insufficient = {1: {0: [0], 1: [4]}}  # total 2 < rem 3
    try:
        sched_pack._select_rsets_for_gpu_group(1, 3, bins_insufficient)
        assert False, "should raise InsufficientFreeResources"
    except InsufficientFreeResources:
        pass
    # Also whole+rem insufficient: whole consumes 4, only 1 left but need 3
    bins_whole_insuf = {1: {0: [0, 1, 2, 3], 1: [4]}}
    try:
        sched_pack._select_rsets_for_gpu_group(1, 7, bins_whole_insuf)
        assert False, "should raise InsufficientFreeResources"
    except InsufficientFreeResources:
        pass


def test_assign_gpu_sharing_resources():
    """Unit + integration coverage for ``_assign_gpu_sharing_resources``.

    Validates the end-to-end physical-exclusive allocator:

    * **Pack co-location:** two consecutive 1-share ``assign_resources``
      calls land on the same ``gpu_id`` and same ``group`` (node).
    * **Spread node selection:** same two calls with ``gpu_packing=spread``
      land on different groups (nodes) because ``feasible`` groups are
      sorted by ``-total_free``.
    * **Exclusive multi-GPU:** 4 shares → 2 distinct physical GPUs on one
      node; 3 shares → 2 distinct GPUs (whole+rem).
    * **Error paths:** requesting 20 shares exceeds total GPU rsets and
      raises ``InsufficientResourcesError``/``InsufficientFreeResources``;
      invalid ``gpu_packing=\"bad\"`` raises ``ValueError``.
    * **GPU groups:** with ``gpus_per_group=2`` (scaled by sharing factor)
      an 8-share request correctly spans the expanded group threshold.
    """
    # Pack co-location: two 1-share tasks should stack on same physical GPU/node
    gres, sched = _setup_resources_sharing(sched_opts={"gpu_packing": "pack"})
    team = sched.assign_resources(rsets_req=1, use_gpus=True)
    assert team == [0]  # first share on GPU 0, group 1
    assert sched.gpu_rsets_free == 15  # one GPU share consumed
    team2 = sched.assign_resources(rsets_req=1, use_gpus=True)
    assert team2 == [1]  # second share packs onto same GPU
    assert sched.resources.rsets[team[0]]["gpu_id"] == sched.resources.rsets[team2[0]]["gpu_id"]
    assert sched.resources.rsets[team[0]]["group"] == sched.resources.rsets[team2[0]]["group"]
    # Spread node selection: same 1-share pair should go to different nodes
    _, sched_spread = _setup_resources_sharing(sched_opts={"gpu_packing": "spread"})
    t1 = sched_spread.assign_resources(rsets_req=1, use_gpus=True)
    t2 = sched_spread.assign_resources(rsets_req=1, use_gpus=True)
    assert sched_spread.resources.rsets[t1[0]]["group"] != sched_spread.resources.rsets[t2[0]]["group"]
    # Exclusive multi-GPU: 4 shares = 2 whole GPUs, must be distinct physical ids on one node
    _, sched_exc = _setup_resources_sharing()  # fresh pack scheduler
    team_exc = sched_exc.assign_resources(rsets_req=4, use_gpus=True)
    assert len(team_exc) == 4
    assert len(set(sched_exc.resources.rsets[team_exc]["gpu_id"])) == 2
    # Whole+remainder: 3 shares = 1 whole +1, also 2 distinct GPUs
    team_frac = sched_exc.assign_resources(rsets_req=3, use_gpus=True)
    assert len(team_frac) == 3
    assert len(set(sched_exc.resources.rsets[team_frac]["gpu_id"])) == 2
    # Error paths: over-request beyond total GPU shares (16) and invalid packing option
    try:
        sched_exc.assign_resources(rsets_req=20, use_gpus=True)
        assert False, "should have raised Insufficient*"
    except Exception as e:
        assert e.__class__.__name__ in ("InsufficientResourcesError", "InsufficientFreeResources")
    try:
        _ = ResourceScheduler(user_resources=gres, sched_opts={"gpu_packing": "bad"})
        assert False, "invalid packing should raise ValueError"
    except ValueError:
        pass  # expected: packing must be 'pack' or 'spread'
    # GPU groups: gpus_per_group scaled by sf, 8 shares should fill one expanded group
    _, sched_grp = _setup_resources_sharing(gpus_per_group=2)
    team_grp = sched_grp.assign_resources(rsets_req=8, use_gpus=True)
    assert len(team_grp) == 8


if __name__ == "__main__":
    test_request_zero_rsets()
    test_too_many_rsets()
    test_cannot_split_quick_return()
    test_schedule_find_gaps_1node()
    test_schedule_find_gaps_2nodes()
    test_split_across_no_matching_slots()
    test_across_nodes_even_split()
    test_across_nodes_roundup_option_2nodes()
    test_across_nodes_roundup_option_3nodes()
    test_try1node_findon_2nodes_matching_slots()
    test_try1node_findon_2nodes_different_slots()
    test_try1node_findon_3nodes()
    test_try2nodes_findon_3nodes()
    test_split2fit_even_required_fails()
    test_split2fit_even_required_various()
    test_try1node_findon_2_or_4nodes()
    test_large_match_slots()
    test_schedule_find_gaps_2nodes_withgpus()
    test_build_gpu_bins()
    test_can_group_satisfy_gpu()
    test_select_rsets_for_gpu_group_pack_spread()
    test_assign_gpu_sharing_resources()
