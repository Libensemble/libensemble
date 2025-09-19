"""
libensemble utility class -- keeps a stack of directory locations.
"""

import os
import shutil
from pathlib import Path


class LocationStack:
    """Keep a stack of directory locations."""

    def __init__(self) -> None:
        """Initialize the location dictionary and directory stack."""
        self.dirs = {}
        self.stack = []

    def copy_file(
        self,
        destdir: Path,
        copy_files: list[Path] = [],
        ignore_FileExists: bool = False,
        allow_overwrite: bool = False,
    ) -> None:
        """Inspired by https://stackoverflow.com/a/9793699.
        Determine paths, basenames, and conditions for copying/symlinking
        """
        for file_path in copy_files:
            file_path = Path(file_path).absolute()
            dest_path = destdir / Path(file_path.name)
            if allow_overwrite and dest_path.exists():
                if dest_path.is_dir():
                    shutil.rmtree(dest_path)
                else:
                    dest_path.unlink()
            try:
                if file_path.is_dir():
                    shutil.copytree(file_path, dest_path, dirs_exist_ok=allow_overwrite)
                else:
                    shutil.copy(file_path, dest_path)
            except FileExistsError:
                if ignore_FileExists:
                    continue
                else:  # Indicates problem with unique sim_dirs
                    raise

    def symlink_file(
        self,
        destdir: Path,
        symlink_files: list[Path] = [],
        ignore_FileExists: bool = False,
        allow_overwrite: bool = False,
    ) -> None:
        for file_path in symlink_files:
            src_path = Path(file_path).absolute()
            dest_path = destdir / Path(file_path.name)
            if allow_overwrite and dest_path.exists():
                dest_path.unlink(missing_ok=True)
            try:
                os.symlink(src_path, dest_path)
            except FileExistsError:
                if ignore_FileExists:
                    continue
                else:  # Indicates problem with unique sim_dirs
                    raise

    def register_loc(
        self,
        key: str | int,
        dirname: Path,
        prefix: Path | None = None,
        copy_files: list[Path] = [],
        symlink_files: list[Path] = [],
        ignore_FileExists: bool = False,
        allow_overwrite: bool = False,
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
        if len(copy_files):
            self.copy_file(dirname, copy_files, ignore_FileExists, allow_overwrite)

        if len(symlink_files):
            self.symlink_file(dirname, symlink_files, ignore_FileExists, allow_overwrite)

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
