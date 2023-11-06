import pydantic

if pydantic.__version__[0] == "1":
    import libensemble.resources.platformsV1 as platforms  # noqa: F401
elif pydantic.__version__[0] == "2":
    import libensemble.resources.platformsV2 as platforms  # noqa: F401
