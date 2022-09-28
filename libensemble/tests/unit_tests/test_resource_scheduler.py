import sys
import time
import pytest
import numpy as np
from libensemble.resources.scheduler import ResourceScheduler, InsufficientFreeResources, InsufficientResourcesError


class MyResources:
    """Simulate resources"""

    rset_dtype = [
        ("assigned", int),  # Holds worker ID assigned to or zero
        ("group", int),  # Group ID this resource set belongs to
        ("slot", int),  # Slot ID this resource set belongs to
    ]

    # Basic layout
    def __init__(self, num_rsets, num_groups):
        self.total_num_rsets = num_rsets
        self.num_groups = num_groups
        self.rsets_per_node = self.total_num_rsets // num_groups
        self.even_groups = True
        self.rsets = np.zeros(self.total_num_rsets, dtype=MyResources.rset_dtype)
        self.rsets["assigned"] = 0
        for i in range(self.total_num_rsets):
            self.rsets["group"][i] = i // self.rsets_per_node
            self.rsets["slot"][i] = i % self.rsets_per_node
        self.rsets_free = self.total_num_rsets

    def free_rsets(self, worker=None):
        """Free up assigned resource sets"""
        if worker is None:
            self.rsets["assigned"] = 0
            self.rsets_free = self.total_num_rsets
        else:
            for rset, wid in enumerate(self.rsets["assigned"]):
                if wid == worker:
                    self.rsets["assigned"][rset] = 0
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
        self.rsets_free = np.count_nonzero(self.rsets["assigned"] == 0)


def _fail_to_resource(sched, rsets):
    with pytest.raises(InsufficientFreeResources):
        rset_team = sched.assign_resources(rsets_req=rsets)
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


def test_too_many_rsets():
    """Tests request of more resource sets than exist"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    resources = MyResources(8, 2)

    # No options
    sched = ResourceScheduler(user_resources=resources)

    with pytest.raises(InsufficientResourcesError):
        rset_team = sched.assign_resources(rsets_req=10)  # noqa F841
        pytest.fail("Expected InsufficientResourcesError")

    del sched
    rset_team = None

    # Options should make no difference
    for match_slots in [False, True]:
        for split2fit in [False, True]:
            sched_options = {"match_slots": match_slots, "split2fit": split2fit}
            sched = ResourceScheduler(user_resources=resources, sched_opts=sched_options)
            with pytest.raises(InsufficientResourcesError):
                rset_team = sched.assign_resources(rsets_req=10)  # noqa F841
                pytest.fail("Expected InsufficientResourcesError")
            del sched
    del resources


def test_cannot_split_quick_return():
    """Tests the quick return when splitting finds no free even gaps"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    resources = MyResources(6, 3)
    resources.fixed_assignment(([1, 0, 0, 0, 3, 3]))
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
            resources.fixed_assignment(([3, 3, 0, 0, 0, 4, 4, 0]))

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
        resources.fixed_assignment(([0, 1, 1, 0, 0, 1]))
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
        ([1, 1, 0, 0, 3, 3, 0, 0]),
        ([0, 1, 0, 2, 0, 4, 0, 4]),
        ([0, 1, 1, 0, 0, 2, 2, 0]),
        ([1, 0, 1, 0, 3, 0, 3, 0]),
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
        ([1, 1, 0, 0, 0, 2, 2, 0]),
        ([1, 1, 0, 0, 0, 0, 3, 3]),
        ([1, 0, 0, 1, 0, 3, 0, 3]),
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
    resources.fixed_assignment(([1, 1, 0, 0, 0, 2, 2, 0, 3, 0, 3, 3]))
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
    resources.fixed_assignment(([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3]))
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
    resources.fixed_assignment(([1, 1, 1, 0, 2, 2, 0, 0]))

    for match_slots in [False, True]:
        sched_options = {"match_slots": match_slots}
        sched = ResourceScheduler(user_resources=resources, sched_opts=sched_options)
        _fail_to_resource(sched, 4)
        assert sched.rsets_free == 3


def test_split2fit_even_required_various():
    """Tests trying to fit to an non-even partition, and setting of local rsets_free"""
    print(f"\nTest: {sys._getframe().f_code.co_name}\n")
    resources = MyResources(8, 2)
    resources.fixed_assignment(([1, 1, 1, 0, 0, 0, 0, 0]))
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
    resources.fixed_assignment(([1, 1, 0, 1, 2, 2, 0, 0, 1, 0, 0, 1, 0, 4, 0, 4]))

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
    rsets["assigned"][col15] = 0

    # Now make sure two rows (groups) with 8, but different slots
    free_row0 = (rsets["group"] == 0) & (rsets["slot"] < 8)
    free_row1 = (rsets["group"] == 1) & (rsets["slot"] >= 8)

    # Now make sure 4 rows of 4 exist (diff slots)
    free_row2 = (rsets["group"] == 2) & (rsets["slot"] < 4)
    free_row3 = (rsets["group"] == 3) & (rsets["slot"] >= 8) & (rsets["slot"] < 12)

    # Now make sure 8 rows of 2 exist (diff slots)
    # Free one slot each as last column already free
    free_strip = (rsets["group"] >= 12) & (rsets["slot"] == 3)

    rsets["assigned"][free_row0] = 0
    rsets["assigned"][free_row1] = 0
    rsets["assigned"][free_row2] = 0
    rsets["assigned"][free_row3] = 0
    rsets["assigned"][free_strip] = 0

    resources.free_rsets = np.count_nonzero(rsets["assigned"] == 0)
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
