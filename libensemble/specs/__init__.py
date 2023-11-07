from libensemble.utils.pydantic_support import pydanticV1, pydanticV2

if pydanticV1:
    from .specsV1 import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs, _EnsembleSpecs  # noqa: F401
elif pydanticV2:
    from .specsV2 import AllocSpecs, ExitCriteria, GenSpecs, LibeSpecs, SimSpecs, _EnsembleSpecs  # noqa: F401
