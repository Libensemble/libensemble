import pydantic

pydantic_version = pydantic.__version__[0]

pydanticV1 = pydantic_version == "1"
pydanticV2 = pydantic_version == "2"


def specs_dump(specs, **kwargs):
    if pydantic_version == "1":
        return specs.dict(**kwargs)
    elif pydantic_version == "2":
        return specs.model_dump(**kwargs)
