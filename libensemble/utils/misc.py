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


def _decide_dtype(name: str, entry, size: int) -> tuple:
    if isinstance(entry, str):
        output_type = "U" + str(len(entry) + 1)
    else:
        output_type = type(entry)
    if size == 1 or not size:
        return (name, output_type)
    else:
        return (name, output_type, (size,))


def _combine_names(names: list) -> list:
    """combine fields with same name *except* for final digits"""

    out_names = []
    stripped = list(i.rstrip("0123456789") for i in names)  # ['x', 'x', y', 'z', 'a']
    for name in names:
        stripped_name = name.rstrip("0123456789")
        if stripped.count(stripped_name) > 1:  # if name appears >= 1, will combine, don't keep int suffix
            out_names.append(stripped_name)
        else:
            out_names.append(name)  # name appears once, keep integer suffix, e.g. "co2"

    # intending [x, y, z, a0] from [x0, x1, y, z0, z1, z2, z3, a0]
    return list(set(out_names))


def list_dicts_to_np(list_dicts: list, dtype: list = None) -> npt.NDArray:
    if list_dicts is None:
        return None

    if not isinstance(list_dicts, list):  # presumably already a numpy array, conversion not necessary
        return list_dicts

    for entry in list_dicts:
        if "_id" in entry:
            entry["sim_id"] = entry.pop("_id")

    first = list_dicts[0]  # for determining dtype of output np array
    new_dtype_names = _combine_names([i for i in first.keys()])  # -> ['x', 'y']
    combinable_names = []  # [['x0', 'x1'], ['y0', 'y1', 'y2'], ['z']]
    for name in new_dtype_names:  # is this a necessary search over the keys again? we did it earlier...
        combinable_group = [i for i in first.keys() if i.rstrip("0123456789") == name]
        if len(combinable_group) > 1:  # multiple similar names, e.g. x0, x1
            combinable_names.append(combinable_group)
        else:  # single name, e.g. local_pt, a0 *AS LONG AS THERE ISNT AN A1*
            combinable_names.append([name])

    if dtype is None:
        dtype = []

    if not len(dtype):
        # another loop over names, there's probably a more elegant way, but my brain is fried
        for i, entry in enumerate(combinable_names):
            name = new_dtype_names[i]
            size = len(combinable_names[i])
            dtype.append(_decide_dtype(name, first[entry[0]], size))

    out = np.zeros(len(list_dicts), dtype=dtype)

    for i, group in enumerate(combinable_names):
        new_dtype_name = new_dtype_names[i]
        for j, input_dict in enumerate(list_dicts):
            if len(group) == 1:  # only a single name, e.g. local_pt
                out[new_dtype_name][j] = input_dict[new_dtype_name]
            else:  # combinable names detected, e.g. x0, x1
                out[new_dtype_name][j] = tuple([input_dict[name] for name in group])

    return out


def np_to_list_dicts(array: npt.NDArray) -> List[dict]:
    if array is None:
        return None
    out = []
    for row in array:
        new_dict = {}
        for field in row.dtype.names:
            # non-string arrays, lists, etc.
            if hasattr(row[field], "__len__") and len(row[field]) > 1 and not isinstance(row[field], str):
                for i, x in enumerate(row[field]):
                    new_dict[field + str(i)] = x
            elif hasattr(row[field], "__len__") and len(row[field]) == 1:  # single-entry arrays, lists, etc.
                new_dict[field] = row[field][0]  # will still work on single-char strings
            else:
                new_dict[field] = row[field]
        out.append(new_dict)

    for entry in out:
        if "sim_id" in entry:
            entry["_id"] = entry.pop("sim_id")

    return out
