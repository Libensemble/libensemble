"""
Misc internal functions
"""

from itertools import groupby
from operator import itemgetter

import pydantic

pydantic_version = pydantic.__version__[0]

pydanticV1 = pydantic_version == "1"
pydanticV2 = pydantic_version == "2"

if not pydanticV1 and not pydanticV2:
    raise ModuleNotFoundError("Pydantic not installed or current version not supported. Install v1 or v2.")


def extract_H_ranges(Work: dict) -> str:
    """Convert received H_rows into ranges for labeling"""
    work_H_rows = Work["libE_info"]["H_rows"]
    if len(work_H_rows) == 1:
        return str(work_H_rows[0])
    else:
        # From https://stackoverflow.com/a/30336492
        ranges = []
        for diff, group in groupby(enumerate(work_H_rows.tolist()), lambda x: x[0] - x[1]):
            group = list(map(itemgetter(1), group))
            if len(group) > 1:
                ranges.append(str(group[0]) + "-" + str(group[-1]))
            else:
                ranges.append(str(group[0]))
        return "_".join(ranges)


class _WorkerIndexer:
    def __init__(self, iterable: list, additional_worker=False):
        self.iterable = iterable
        self.additional_worker = additional_worker

    def __getitem__(self, key):
        if self.additional_worker or isinstance(key, str):
            return self.iterable[key]
        else:
            return self.iterable[key - 1]

    def __setitem__(self, key, value):
        self.iterable[key] = value

    def __len__(self):
        return len(self.iterable)

    def __iter__(self):
        return iter(self.iterable)


def specs_dump(specs, **kwargs):
    if pydanticV1:
        return specs.dict(**kwargs)
    else:
        return specs.model_dump(**kwargs)


def specs_checker_getattr(obj, key, default=None):
    if pydanticV1:  # dict
        return obj.get(key, default)
    else:  # actual obj
        try:
            return getattr(obj, key)
        except AttributeError:
            return default


def specs_checker_setattr(obj, key, value):
    if pydanticV1:  # dict
        obj[key] = value
    else:  # actual obj
        obj.__dict__[key] = value
