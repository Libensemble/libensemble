import os
import re
import shutil
from dataclasses import dataclass

from libensemble.message_numbers import EVAL_SIM_TAG, calc_type_strings
from libensemble.tools.fields_keys import (libE_spec_calc_dir_misc,
                                           libE_spec_gen_dir_keys,
                                           libE_spec_sim_dir_keys)
from libensemble.utils.loc_stack import LocationStack
from libensemble.utils.misc import extract_H_ranges


class EnsembleDirectory:
    """
    The EnsembleDirectory class provides methods for workers to initialize and
    manipulate the optional output directories where workers can change to before
    calling user functions.

    The top-level ensemble directory typically stores unique sub-directories containing results
    for each libEnsemble user function call. This can be a separate location
    on other filesystems or directories (like scratch spaces).

    When libEnsemble is initialized in a Distributed fashion, each worker can
    initiate its own ensemble directory on the local node, and copy
    back its results on completion or exception into the directory that libEnsemble
    was originally launched from.

    Ensemble directory behavior can be configured via separate libE_specs
    dictionary entries or defining an EnsembleDirectory object within libE_specs.

    Parameters
    ----------
    libE_specs: dict
        Parameters/information for libE operations. EnsembleDirectory only extracts
        values specific for ensemble directory operations. Can technically contain
        a different set of settings then the libE_specs passed to libE().

    loc_stack: object
        A LocationStack object from libEnsemble's internal libensemble.utils.loc_stack module.
    """

    def __init__(self, libE_specs=None, workerID=None, loc_stack=None):

        self.specs = libE_specs
        self.workerID = workerID
        self.loc_stack = loc_stack

        if self.specs is not None:
            self.prefix = self.specs.get("ensemble_dir_path", "./ensemble")
            self.use_worker_dirs = self.specs.get("use_worker_dirs", False)
            self.sim_input_dir = self.specs.get("sim_input_dir", "").rstrip("/")
            self.sim_dirs_make = self.specs.get("sim_dirs_make", False)
            self.sim_dir_copy_files = self.specs.get("sim_dir_copy_files", [])
            self.sim_dir_symlink_files = self.specs.get("sim_dir_symlink_files", [])
            self.gen_input_dir = self.specs.get("gen_input_dir", "").rstrip("/")
            self.gen_dirs_make = self.specs.get("gen_dirs_make", False)
            self.gen_dir_copy_files = self.specs.get("gen_dir_copy_files", [])
            self.gen_dir_symlink_files = self.specs.get("gen_dir_symlink_files", [])
            self.ensemble_copy_back = self.specs.get("ensemble_copy_back", False)
            self.sim_use = any([self.specs.get(i) for i in libE_spec_sim_dir_keys + libE_spec_calc_dir_misc])
            self.gen_use = any([self.specs.get(i) for i in libE_spec_gen_dir_keys + libE_spec_calc_dir_misc])

    def _make_copyback_dir(self):
        """Make copyback directory, adding suffix if identical to ensemble dir"""
        copybackdir = os.path.basename(self.prefix)  # Current directory, same basename
        if os.path.relpath(self.prefix) == os.path.relpath(copybackdir):
            copybackdir += "_back"
        os.makedirs(copybackdir)

    def make_copyback_check(self):
        """Check for existing copyback, make copyback if doesn't exist"""
        try:
            os.rmdir(self.prefix)
        except FileNotFoundError:
            pass
        except Exception:
            raise
        if self.ensemble_copy_back:
            self._make_copyback_dir()

    def use_calc_dirs(self, intype):
        """Determines calc_dirs enabling for each calc type"""
        if intype == EVAL_SIM_TAG:
            return self.sim_use
        else:
            return self.gen_use

    def _make_calc_dir(self, H_rows, calc_str):
        """Create calc dirs and intermediate dirs, copy inputs, based on libE_specs"""
        if calc_str == "sim":
            input_dir = self.sim_input_dir
            do_calc_dirs = self.sim_dirs_make
            copy_files = self.sim_dir_copy_files
            symlink_files = self.sim_dir_symlink_files
        else:  # calc_str is 'gen'
            input_dir = self.gen_input_dir
            do_calc_dirs = self.gen_dirs_make
            copy_files = self.gen_dir_copy_files
            symlink_files = self.gen_dir_symlink_files

        # If 'use_worker_dirs' only calc_dir option. Use worker dirs, but no calc dirs
        if self.use_worker_dirs and not self.sim_dirs_make and not self.gen_dirs_make:
            do_calc_dirs = False

        # If using input_dir, set of files to copy is contents of provided dir
        if input_dir:
            copy_files = set(copy_files + [os.path.join(input_dir, i) for i in os.listdir(input_dir)])

        # If identical paths to copy and symlink, remove those paths from symlink_files
        if len(symlink_files):
            symlink_files = [i for i in symlink_files if i not in copy_files]

        # Cases where individual sim_dirs or gen_dirs not created.
        if not do_calc_dirs:
            if self.use_worker_dirs:  # Each worker does work in worker dirs
                key = self.workerID
                cdir = "worker" + str(self.workerID)
                prefix = self.prefix
            else:  # Each worker does work in prefix (ensemble_dir)
                key = self.prefix
                cdir = self.prefix
                prefix = None

            self.locs.register_loc(
                key,
                cdir,
                prefix=prefix,
                copy_files=copy_files,
                symlink_files=symlink_files,
                ignore_FileExists=True,
            )
            return key

        # All cases now should involve sim_dirs or gen_dirs
        # ensemble_dir/worker_dir registered here, set as parent dir for calc dirs
        if self.use_worker_dirs:
            worker_dir = "worker" + str(self.workerID)
            worker_path = os.path.abspath(os.path.join(self.prefix, worker_dir))
            calc_dir = calc_str + str(H_rows)
            self.locs.register_loc(self.workerID, worker_dir, prefix=self.prefix)
            calc_prefix = worker_path

        # Otherwise, ensemble_dir set as parent dir for sim dirs
        else:
            calc_dir = f"{calc_str}{H_rows}_worker{self.workerID}"
            if not os.path.isdir(self.prefix):
                os.makedirs(self.prefix, exist_ok=True)
            calc_prefix = self.prefix

        # Register calc dir with adjusted parent dir and source-file location
        self.locs.register_loc(
            calc_dir,
            calc_dir,  # Dir name also label in loc stack dict
            prefix=calc_prefix,
            copy_files=copy_files,
            symlink_files=symlink_files,
        )

        return calc_dir

    def prep_calc_dir(self, Work, calc_iter, calc_type):
        """Determines choice for calc_dir structure, then performs calculation."""
        if not self.loc_stack:
            self.loc_stack = LocationStack()

        if calc_type == EVAL_SIM_TAG:
            H_rows = extract_H_ranges(Work)
        else:
            H_rows = str(calc_iter[calc_type])

        calc_str = calc_type_strings[calc_type]
        calc_dir = self._make_calc_dir(H_rows, calc_str)

        return self.loc_stack, calc_dir

    def copy_back(self):
        """Copy back all ensemble dir contents to launch location"""
        if os.path.isdir(self.prefix) and self.ensemble_copy_back:

            copybackdir = os.path.basename(self.prefix)

            if os.path.relpath(self.prefix) == os.path.relpath(copybackdir):
                copybackdir += "_back"

            for cdir in self.loc_stack.dirs.values():
                dest_path = os.path.join(copybackdir, os.path.basename(dir))
                if cdir == self.prefix:
                    continue  # otherwise, entire ensemble dir copied into copyback dir

                shutil.copytree(cdir, dest_path, symlinks=True)
                if os.path.basename(cdir).startswith("worker"):
                    return  # Worker dir (with all contents) has been copied.

            # If not using calc dirs, likely miscellaneous files to copy back
            if not self.sim_dirs_make or not self.gen_dirs_make:
                p = re.compile(r"((^sim)|(^gen))\d+_worker\d+")
                for file in [i for i in os.listdir(self.prefix) if not p.match(i)]:  # each non-calc_dir file
                    source_path = os.path.join(self.prefix, file)
                    dest_path = os.path.join(copybackdir, file)
                    try:
                        if os.path.isdir(source_path):
                            shutil.copytree(source_path, dest_path, symlinks=True)
                        else:
                            shutil.copy(source_path, dest_path, follow_symlinks=False)
                    except FileExistsError:
                        continue
                    except shutil.SameFileError:  # creating an identical symlink
                        continue
