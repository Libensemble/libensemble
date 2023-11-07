from libensemble.utils.pydantic_support import pydanticV1, pydanticV2

if pydanticV1:
    from .platforms import platformsV1 as platforms  # noqa: F401
elif pydanticV2:
    from .platforms import platformsV2 as platforms  # noqa: F401
