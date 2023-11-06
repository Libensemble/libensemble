import pydantic

if pydantic.__version__[0] == "1":
    from .platforms import platformsV1 as platforms  # noqa: F401
elif pydantic.__version__[0] == "2":
    from .platforms import platformsV2 as platforms  # noqa: F401
