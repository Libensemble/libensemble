import pydantic

pydantic_version = pydantic.__version__[0]

pydanticV1 = pydantic_version == "1"
pydanticV2 = pydantic_version == "2"

if pydanticV1:
    from pydantic import root_validator, validator
elif pydanticV2:
    from pydantic import field_validator, model_validator


def specs_dump(specs, **kwargs):
    if pydantic_version == "1":
        return specs.dict(**kwargs)
    elif pydantic_version == "2":
        return specs.model_dump(**kwargs)


def specs_field_validate(func, *args):
    if pydantic_version == "1":
        return validator(*args)(func)
    elif pydantic_version == "2":
        return classmethod(field_validator(*args)(func))


def specs_model_validate(func):
    if pydantic_version == "1":
        return root_validator(func)
    elif pydantic_version == "2":
        return model_validator(mode="after")(func)
