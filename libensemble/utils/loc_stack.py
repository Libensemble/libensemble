"""
libensemble utility class -- keeps a stack of directory locations.
"""

import os
import shutil


class LocationStack:
    """Keep a stack of directory locations."""

    def __init__(self):
        """Initialize the location dictionary and directory stack."""
        self.dirs = {}
        self.stack = []

    def copy_or_symlink(self, destdir, copy_files=[], symlink_files=[], ignore_FileExists=False):
        """Inspired by https://stackoverflow.com/a/9793699.
        Determine paths, basenames, and conditions for copying/symlinking
        """
        for file_path in copy_files:
            file_path = os.path.expanduser(os.path.expandvars(file_path))
            src_base = os.path.basename(file_path)
            dest_path = os.path.join(destdir, src_base)
            try:
                if os.path.isdir(file_path):
                    shutil.copytree(file_path, dest_path)
                else:
                    shutil.copy(file_path, dest_path)
            except FileExistsError:
                if ignore_FileExists:
                    continue
                else:  # Indicates problem with unique sim_dirs
                    raise

        for file_path in symlink_files:
            src_path = os.path.abspath(os.path.expanduser(os.path.expandvars(file_path)))
            dest_path = os.path.join(destdir, os.path.basename(file_path))
            try:
                os.symlink(src_path, dest_path)
            except FileExistsError:
                if ignore_FileExists:
                    continue
                else:  # Indicates problem with unique sim_dirs
                    raise

    def register_loc(self, key, dirname, prefix=None, copy_files=[], symlink_files=[], ignore_FileExists=False):
        """Register a new location in the dictionary.

        Parameters
        ----------

        key:
            The key used to identify the new location.

        dirname: string:
            Directory name

        prefix: string:
            Prefix to be used with the dirname.  If prefix is not None,
            only the base part of the dirname is used.

        copy_files: list:
            Copy these files to the destination directory.

        symlink_files: list:
            Symlink these files to the destination directory.
        """
        if prefix is not None:
            prefix = os.path.expanduser(prefix)
            dirname = os.path.join(prefix, os.path.basename(dirname))

        if dirname and not os.path.isdir(dirname):
            os.makedirs(dirname, exist_ok=True)  # Prevent race-condition when no sim_dirs

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
            self.stack.append(os.getcwd())
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
