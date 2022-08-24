# From https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess

# Usage:
# from libensemble.tools import ForkablePdb
# ForkablePdb().set_trace()

import pdb
import sys


# Best implementation depends on system.
# If one of these does not work, try the other.
class ForkablePdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    Usage:

    .. code-block:: python

        from libensemble.tools import ForkablePdb
        ForkablePdb().set_trace()

    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


# class ForkablePdb(pdb.Pdb):
#     """A Pdb subclass that may be used
#     #from a forked multiprocessing child"""
#     _original_stdin_fd = sys.stdin.fileno()
#     _original_stdin = None

#     def __init__(self):
#         pdb.Pdb.__init__(self, nosigint=True)

#     def _cmdloop(self):
#         current_stdin = sys.stdin
#         try:
#             if not self._original_stdin:
#                 self._original_stdin = os.fdopen(self._original_stdin_fd)
#             sys.stdin = self._original_stdin
#             self.cmdloop()
#         finally:
#             sys.stdin = current_stdin
