import pydantic

if pydantic.__version__[0] == "1":
    from .specsV1 import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs, _EnsembleSpecs  # noqa: F401
elif pydantic.__version__[0] == "2":
    from .specsV2 import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs, _EnsembleSpecs  # noqa: F401
