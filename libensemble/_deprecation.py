"""
Deprecation utilities for libEnsemble.
"""

import warnings


class LibEnsembleDeprecationWarning(DeprecationWarning):
    """Warning category for deprecated libEnsemble features.

    Subclass of :class:`DeprecationWarning` so users can filter libEnsemble
    deprecations independently::

        import warnings
        from libensemble._deprecation import LibEnsembleDeprecationWarning
        warnings.filterwarnings("error", category=LibEnsembleDeprecationWarning)
    """


def warn_deprecated(name: str, replacement: str, removal_version: str = "2.1") -> None:
    """Emit a :class:`LibEnsembleDeprecationWarning` for a deprecated feature.

    Parameters
    ----------
    name:
        Dotted module or object path (e.g. ``"libensemble.alloc_funcs.fast_alloc"``).
    replacement:
        Human-readable description of the recommended replacement.
    removal_version:
        The libEnsemble version in which the feature will be removed.
    """
    warnings.warn(
        f"{name} is deprecated as of libEnsemble 2.0 "
        f"and will be removed in {removal_version}. "
        f"Use {replacement} instead.",
        LibEnsembleDeprecationWarning,
        stacklevel=3,
    )
