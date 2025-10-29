import os
import re
import shutil
from pathlib import Path

from libensemble.message_numbers import EVAL_SIM_TAG, calc_type_strings
from libensemble.tools.fields_keys import libE_spec_calc_dir_misc, libE_spec_gen_dir_keys, libE_spec_sim_dir_keys
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

    def __init__(self, libE_specs: dict, loc_stack: LocationStack | None = None):
        self.specs = libE_specs
        self.loc_stack = loc_stack

        self.ensemble_dir = Path(self.specs.get("ensemble_dir_path", "ensemble"))
        self.workflow_dir = Path(self.specs.get("workflow_dir_path", ""))
        self.use_worker_dirs = self.specs.get("use_worker_dirs", False)
        self.ensemble_copy_back = self.specs.get("ensemble_copy_back", False)
        self.allow_overwrite = self.specs.get("reuse_output_dir", False)

        self.sim_use = any([self.specs.get(i) for i in libE_spec_sim_dir_keys + libE_spec_calc_dir_misc])
        self.sim_input_dir = Path(self.specs.get("sim_input_dir")) if self.specs.get("sim_input_dir") else ""
        self.sim_dirs_make = self.specs.get("sim_dirs_make", False)
        self.sim_dir_copy_files = self.specs.get("sim_dir_copy_files", [])
        self.sim_dir_symlink_files = self.specs.get("sim_dir_symlink_files", [])

        self.gen_use = any([self.specs.get(i) for i in libE_spec_gen_dir_keys + libE_spec_calc_dir_misc])
        self.gen_input_dir = Path(self.specs.get("gen_input_dir")) if self.specs.get("gen_input_dir") else ""
        self.gen_dirs_make = self.specs.get("gen_dirs_make", False)
        self.gen_dir_copy_files = self.specs.get("gen_dir_copy_files", [])
        self.gen_dir_symlink_files = self.specs.get("gen_dir_symlink_files", [])

        if self.workflow_dir and self.ensemble_dir.stem == "ensemble":  # default ensemble dir without adjustment
            self.ensemble_dir = self.workflow_dir / self.ensemble_dir

        if self.ensemble_copy_back:
            self.copybackdir = self.workflow_dir / Path(self.ensemble_dir.stem + "_back")

        self.pad = self.specs.get("calc_dir_id_width", 4)

    def make_copyback(self) -> None:
        """Check for existing ensemble dir and copybackdir, make copyback if doesn't exist"""
        try:
            assert not self.ensemble_dir.exists()
        except AssertionError:
            if not self.allow_overwrite:
                raise
        except Exception:
            raise
        if self.ensemble_copy_back:
            if not self.allow_overwrite:
                self.copybackdir.mkdir()
            else:
                self.copybackdir.mkdir(exist_ok=True)

    def use_calc_dirs(self, typelabel: int) -> bool:
        """Determines calc_dirs enabling for each calc type"""
        if typelabel == EVAL_SIM_TAG:
            return self.sim_use
        else:
            return self.gen_use

    def _make_calc_dir(self, workerID, H_rows, calc_str: str, locs: LocationStack):
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
            copy_files = set(copy_files + [i for i in input_dir.iterdir()])

        # If identical paths to copy and symlink, remove those paths from symlink_files
        if len(symlink_files):
            symlink_files = [i for i in symlink_files if i not in copy_files]

        # Cases where individual sim_dirs or gen_dirs not created.
        if not do_calc_dirs:
            if self.use_worker_dirs:  # Each worker does work in worker dirs
                key = workerID
                dirname = "worker" + str(workerID)
                prefix = self.ensemble_dir
            else:  # Each worker does work in prefix (ensemble_dir)
                key = self.ensemble_dir
                dirname = self.ensemble_dir
                prefix = None

            locs.register_loc(
                key,
                Path(dirname),
                prefix=prefix,
                copy_files=copy_files,
                symlink_files=symlink_files,
                ignore_FileExists=True,
                allow_overwrite=self.allow_overwrite,
            )
            return key

        # All cases now should involve sim_dirs or gen_dirs
        # ensemble_dir/worker_dir registered here, set as parent dir for calc dirs

        if self.use_worker_dirs:
            worker_dir = "worker" + str(workerID)
            worker_path = (self.ensemble_dir / Path(worker_dir)).absolute()
            locs.register_loc(
                workerID, Path(worker_dir), prefix=self.ensemble_dir, allow_overwrite=self.allow_overwrite
            )
            calc_prefix = worker_path

        # Otherwise, ensemble_dir set as parent dir for sim dirs
        else:
            if not self.ensemble_dir.exists():
                self.ensemble_dir.mkdir(exist_ok=True, parents=True)
            calc_prefix = self.ensemble_dir

        calc_dir = calc_str + str(H_rows).rjust(self.pad, str(0))
        # Register calc dir with adjusted parent dir and sourcefile location
        locs.register_loc(
            calc_dir,
            Path(calc_dir),  # Dir name also label in loc stack dict
            prefix=calc_prefix,
            copy_files=copy_files,
            symlink_files=symlink_files,
            allow_overwrite=self.allow_overwrite,
        )

        return calc_dir

    def prep_calc_dir(self, Work: dict, calc_iter: dict, workerID: int, calc_type: int) -> (LocationStack, str):
        """Determines choice for calc_dir structure, then performs calculation."""
        if not self.loc_stack:
            self.loc_stack = LocationStack()

        if calc_type == EVAL_SIM_TAG:
            H_rows = extract_H_ranges(Work)
        else:
            H_rows = str(calc_iter[calc_type])

        calc_str = calc_type_strings[calc_type]

        calc_dir = self._make_calc_dir(workerID, H_rows, calc_str, self.loc_stack)

        return self.loc_stack, calc_dir

    def copy_back(self) -> None:
        """Copy back all ensemble dir contents to launch location"""
        if not self.ensemble_dir.exists() or not self.ensemble_copy_back or not self.loc_stack:
            return

        for dire in self.loc_stack.dirs.values():
            dire = Path(dire)
            dest_path = self.copybackdir / Path(dire.stem)
            if dire == self.ensemble_dir:  # occurs when no_calc_dirs is True
                continue  # otherwise, entire ensemble dir copied into copyback dir

            if self.allow_overwrite:
                shutil.rmtree(dest_path, ignore_errors=True)
            shutil.copytree(dire, dest_path, symlinks=True, dirs_exist_ok=True)
            if dire.stem.startswith("worker"):
                return  # Worker dir (with all contents) has been copied.

        # If not using calc dirs, likely miscellaneous files to copy back
        if not self.sim_dirs_make or not self.gen_dirs_make:
            p = re.compile(r"((^sim)|(^gen))\d")
            for filep in [i for i in os.listdir(self.ensemble_dir) if not p.match(i)]:  # each noncalc_dir file
                source_path = self.ensemble_dir / filep
                dest_path = self.copybackdir / filep
                if self.allow_overwrite:
                    shutil.rmtree(dest_path, ignore_errors=True)
                try:
                    if os.path.isdir(source_path):
                        shutil.copytree(source_path, dest_path, symlinks=True)
                    else:
                        shutil.copy(source_path, dest_path, follow_symlinks=False)
                except (FileExistsError, shutil.SameFileError):  # creating an identical symlink
                    continue
