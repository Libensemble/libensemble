"""
libensemble utility class -- keeps a stack of directory locations.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Union


class LocationStack:
    """Keep a stack of directory locations."""

    def __init__(self) -> None:
        """Initialize the location dictionary and directory stack."""
        self.dirs = {}
        self.stack = []

    def copy_or_symlink(
        self, destdir: str, copy_files: List[Path] = [], symlink_files: List[Path] = [], ignore_FileExists: bool = False
    ) -> None:
        """Inspired by https://stackoverflow.com/a/9793699.
        Determine paths, basenames, and conditions for copying/symlinking
        """
        for file_path in copy_files:
            file_path = Path(file_path).absolute()
            dest_path = destdir / Path(file_path.name)
            try:
                if file_path.is_dir():
                    shutil.copytree(file_path, dest_path)
                else:
                    shutil.copy(file_path, dest_path)
            except FileExistsError:
                if ignore_FileExists:
                    continue
                else:  # Indicates problem with unique sim_dirs
                    raise

        for file_path in symlink_files:
            src_path = Path(file_path).absolute()
            dest_path = destdir / Path(file_path.name)
            try:
                os.symlink(src_path, dest_path)
            except FileExistsError:
                if ignore_FileExists:
                    continue
                else:  # Indicates problem with unique sim_dirs
                    raise

    def register_loc(
        self,
        key: Union[str, int],
        dirname: Path,
        prefix: Optional[Path] = None,
        copy_files: List[Path] = [],
        symlink_files: List[Path] = [],
        ignore_FileExists: bool = False,
    ) -> str:
        """Register a new location in the dictionary.

        Parameters
        ----------

        key:
            The key used to identify the new location.

        dirname: Path:
            Directory name

        prefix: Path:
            Prefix to be used with the dirname.  If prefix is not None,
            only the base part of the dirname is used.

        copy_files: list of Paths:
            Copy these files to the destination directory.

        symlink_files: list of Paths:
            Symlink these files to the destination directory.
        """
        if prefix is not None:
            dirname = prefix.absolute() / dirname.stem

        if dirname and not dirname.is_dir():
            dirname.mkdir(parents=True, exist_ok=True)

        self.dirs[key] = dirname
        if len(copy_files) or len(symlink_files):
            self.copy_or_symlink(dirname, copy_files, symlink_files, ignore_FileExists)

        return dirname

    def push_loc(self, key):
        """Push a location from the dictionary."""
        self.push(self.dirs.get(key))

    def clean_locs(self):
        """Remove all directories listed in the dictionary."""
        for dirname in self.dirs.values():
            if dirname is not None and os.path.isdir(dirname):
                shutil.rmtree(dirname)

    def push(self, dirname):
        """Push the current location and change directories (if not None)."""
        if dirname is not None:
            self.stack.append(Path.cwd())
            os.chdir(dirname)
        else:
            self.stack.append(None)

    def pop(self):
        """Pop the current directory and change back."""
        dirname = self.stack.pop()
        if dirname is not None:
            os.chdir(dirname)

    class Saved:
        """Context object for use with a with statement"""

        def __init__(self, ls, dirname):
            self.ls = ls
            self.dirname = dirname

        def __enter__(self):
            self.ls.push(self.dirname)
            return self.ls

        def __exit__(self, etype, value, traceback):
            self.ls.pop()

    def loc(self, key):
        """Return a with context for pushing a location key"""
        return LocationStack.Saved(self, self.dirs.get(key))

    def dir(self, dirname):
        """Return a with context for pushing a directory"""
        return LocationStack.Saved(self, dirname)
