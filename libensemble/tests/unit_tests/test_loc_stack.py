#!/usr/bin/env python

"""
Unit test of location stack for libensemble.
"""

import os
import shutil
import tempfile

from libensemble.utils.loc_stack import LocationStack


def test_location_stack():
    "Test correctness of location stack (all in a temp dir)."

    tmp_dirname = tempfile.mkdtemp()
    assert os.path.isdir(tmp_dirname), f"Failed to create temporary directory {tmp_dirname}."

    try:
        # Record where we started
        start_dir = os.getcwd()

        # Set up directory for clone
        clone_dirname = os.path.join(tmp_dirname, "basedir")
        os.mkdir(clone_dirname)
        test_fname = os.path.join(clone_dirname, "test.txt")
        with open(test_fname, "w+") as f:
            f.write("This is a test file\n")

        s = LocationStack()

        # Register a valid location
        tname = s.register_loc(0, "testdir", prefix=tmp_dirname, copy_files=[test_fname])
        assert os.path.isdir(tname), f"New directory {tname} was not created."
        assert os.path.isfile(
            os.path.join(tname, "test.txt")
        ), f"New directory {tname} failed to copy test.txt from {clone_dirname}."

        # Register an empty location
        d = s.register_loc(1, None)
        assert d is None, "Dir stack not correctly register None at location 1."

        # Register a dummy location (del should not work)
        d = s.register_loc(2, os.path.join(tmp_dirname, "dummy"))
        assert ~os.path.isdir(d), "Directory stack registration of dummy should not create dir."

        # Push unregistered location (we should not move)
        s.push_loc(3)
        assert s.stack == [None], "Directory stack push_loc(missing) failed to put None on stack."
        assert os.path.samefile(
            os.getcwd(), start_dir
        ), "Directory stack push_loc failed to stay put with input None." "Wanted {}, at {}".format(
            start_dir, os.getcwd()
        )

        # Push registered location (we should move
        s.push_loc(0)
        assert s.stack == [None, start_dir], "Directory stack is incorrect." "Wanted [None, {}], got {}.".format(
            start_dir, s.stack
        )
        assert os.path.samefile(
            os.getcwd(), tname
        ), f"Directory stack push_loc failed to end up at desired dir.Wanted {tname}, at {os.getcwd()}"

        # Pop the registered location
        s.pop()
        assert s.stack == [None], f"Directory stack is incorrect after pop.Wanted [None], got {s.stack}."
        assert os.path.samefile(
            os.getcwd(), start_dir
        ), "Directory stack push_loc failed to stay put with input None." "Wanted {}, at {}".format(
            start_dir, os.getcwd()
        )

        # Context for moving again
        with s.loc(0):
            assert s.stack == [None, start_dir], "Directory stack is incorrect." "Wanted [None, {}], got {}.".format(
                start_dir, s.stack
            )
            assert os.path.samefile(
                os.getcwd(), tname
            ), f"Directory stack push_loc failed to end up at desired dir.Wanted {tname}, at {os.getcwd()}"

        # Check directory after context
        assert s.stack == [None], f"Directory stack is incorrect after ctx.Wanted [None], got {s.stack}."
        assert os.path.samefile(os.getcwd(), start_dir), "Directory looks wrong after ctx." "Wanted {}, at {}".format(
            start_dir, os.getcwd()
        )

        with s.dir(None):
            assert s.stack == [None, None], "Directory stack is incorrect in ctx."
        assert s.stack == [None], "Directory stack is incorrect after ctx."

        # Pop the unregistered location
        s.pop()
        assert not s.stack, f"Directory stack should be empty, actually {s.stack}."
        assert os.path.samefile(
            os.getcwd(), start_dir
        ), "Directory stack push_loc failed to stay put with input None." "Wanted {}, at {}".format(
            start_dir, os.getcwd()
        )

        # Clean up
        s.clean_locs()
        assert not os.path.isdir(tname), f"Directory {tname} should have been removed on cleanup."

    finally:
        shutil.rmtree(tmp_dirname)
