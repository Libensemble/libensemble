from libensemble.utils.pydantic_bindings import pydanticV1, pydanticV2

if pydanticV1:
    from .platformsV1 import Platform  # noqa: F401
    from .platformsV1 import get_platform  # noqa: F401
elif pydanticV2:
    from .platformsV2 import Platform  # noqa: F401
    from .platformsV2 import get_platform  # noqa: F401
