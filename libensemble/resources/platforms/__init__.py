import pydantic

if pydantic.__version__[0] == "1":
    from .platformsV1 import Platform  # noqa: F401
    from .platformsV1 import get_platform  # noqa: F401
elif pydantic.__version__[0] == "2":
    from .platformsV2 import Platform  # noqa: F401
    from .platformsV2 import get_platform  # noqa: F401
