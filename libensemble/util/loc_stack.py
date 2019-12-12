"""
libensemble utility class -- keeps a stack of directory locations.
"""

import os
import shutil
from glob import glob


class LocationStack:
    """Keep a stack of directory locations.
    """

    def __init__(self):
        """Initialize the location dictionary and directory stack."""
        self.dirs = {}
        self.stack = []

    def sim_dir_symlink(self, srcdir, destdir):
        """ Inspired by https://stackoverflow.com/a/9793699 """
        if not os.path.isdir(destdir):
            os.makedirs(destdir)
        for src_path in glob('{}/*'.format(srcdir)):
            relative_path = os.path.relpath(src_path, destdir)
            link_dest_path = os.path.join(destdir, os.path.basename(src_path))
            os.symlink(relative_path, link_dest_path)

    def register_loc(self, key, dirname, prefix=None, srcdir=None, link=False):
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

        srcdir: string:
            Name of a source directory to populate the new location.
            If srcdir is not None, the directory should not yet exist.
            srcdir is not relative to prefix.
        """
        if prefix is not None:
            prefix = os.path.expanduser(prefix)
            dirname = os.path.join(prefix, os.path.basename(dirname))
        self.dirs[key] = dirname
        if srcdir is not None:
            assert ~os.path.isdir(dirname), \
                "Directory {} already exists".format(dirname)
            if link:
                self.sim_dir_symlink(srcdir, dirname)
            else:
                shutil.copytree(srcdir, dirname)
        else:
            if dirname:
                os.mkdir(dirname)
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
