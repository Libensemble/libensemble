"""
Misc internal functions
"""

from itertools import groupby
from operator import itemgetter
from typing import List

import numpy as np
import pydantic
from numpy import typing as npt

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


def _decide_dtype(name, entry, size):
    if size == 1:
        return (name, type(entry))
    else:
        return (name, type(entry), (size,))


def _combine_names(names):
    return list(set(i[:-1] if i[-1].isdigit() else i for i in names))


def list_dicts_to_np(list_dicts: list) -> npt.NDArray:
    if list_dicts is None:
        return None

    first = list_dicts[0]
    new_dtype_names = _combine_names([i for i in first.keys()])
    new_dtype = []
    combinable_names = []
    for name in new_dtype_names:
        combinable_names.append([i for i in first.keys() if i.startswith(name)])

    for i, entry in enumerate(combinable_names):  # must inspect values to get presumptive types
        name = new_dtype_names[i]
        size = len(combinable_names[i])
        new_dtype.append(_decide_dtype(name, first[entry[0]], size))

    out = np.zeros(len(list_dicts), dtype=new_dtype)

    # good lord, this is ugly
    # for names_group_idx, entry in enumerate(combinable_names):
    #     for input_dict in list_dicts:
    #         for l in range(len(input_dict)):
    #             for name_idx, src_key in enumerate(entry):
    #                 out[new_dtype_names[names_group_idx]][name_idx][l] = input_dict[src_key]

    for name in new_dtype_names:
        for i, input_dict in enumerate(list_dicts):
            for j, value in enumerate(input_dict.values()):
                out[name][j][i] = value

    [
        {"x0": -1.3315287487797274, "x1": -1.1102419596798931},
        {"x0": 2.2035749254093417, "x1": -0.04551905560134939},
        {"x0": -1.043550345357007, "x1": -0.853671651707665},
    ]

    return out


def np_to_list_dicts(array: npt.NDArray) -> List[dict]:
    if array is None:
        return None
    out = []
    for row in array:
        new_dict = {}
        for field in row.dtype.names:
            if len(row[field]) > 1:
                for i, x in enumerate(row[field]):
                    new_dict[field + str(i)] = x
            else:
                new_dict[field] = row[field]
        out.append(new_dict)
    return out
