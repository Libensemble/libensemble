"""
Misc internal functions
"""

from itertools import chain, groupby
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


def _get_new_dtype_fields(first: dict, mapping: dict = {}) -> list:
    """build list of fields that will be in the output numpy array"""
    new_dtype_names = _combine_names([i for i in first.keys()])  # -> ['x', 'y']
    fields_to_convert = list(  # combining all mapping lists
        chain.from_iterable(list(mapping.values()))
    )  # fields like ["beam_length", "beam_width"] that will become "x"
    new_dtype_names = [i for i in new_dtype_names if i not in fields_to_convert] + list(
        mapping.keys()
    )  # array dtype needs "x". avoid fields from mapping values since we're converting those to "x"
    return new_dtype_names


def _get_combinable_multidim_names(first: dict, new_dtype_names: list) -> list:
    """inspect the input dict for fields that can be combined (e.g. x0, x1)"""
    combinable_names = []
    for name in new_dtype_names:
        combinable_group = [i for i in first.keys() if i.rstrip("0123456789") == name]
        if len(combinable_group) > 1:  # multiple similar names, e.g. x0, x1
            combinable_names.append(combinable_group)
        else:  # single name, e.g. local_pt, a0 *AS LONG AS THERE ISNT AN A1*
            combinable_names.append([name])
    return combinable_names


def _decide_dtype(name: str, entry, size: int) -> tuple:
    """decide dtype of field, and size if needed"""
    if isinstance(entry, str):  # use numpy style for string type
        output_type = "U" + str(len(entry) + 1)
    else:
        output_type = type(entry)  # use default "python" type
    if size == 1 or not size:
        return (name, output_type)
    else:
        return (name, output_type, (size,))  # 3-tuple for multi-dimensional


def _start_building_dtype(
    first: dict, new_dtype_names: list, combinable_names: list, dtype: list, mapping: dict
) -> list:
    """parse out necessary components of dtype for output numpy array"""
    for i, entry in enumerate(combinable_names):
        name = new_dtype_names[i]
        size = len(combinable_names[i])  # e.g. 2 for [x0, x1]
        if name not in mapping:  # mapping keys are what we're converting *to*
            dtype.append(_decide_dtype(name, first[entry[0]], size))
    return dtype


def _pack_field(input_dict: dict, field_names: list) -> tuple:
    """pack dict data into tuple for slotting into numpy array"""
    # {"x0": 1, "x1": 2} -> (1, 2)
    return tuple(input_dict[name] for name in field_names) if len(field_names) > 1 else input_dict[field_names[0]]


def list_dicts_to_np(list_dicts: list, dtype: list = None, mapping: dict = {}) -> npt.NDArray:
    if list_dicts is None:
        return None

    if not isinstance(list_dicts, list):  # presumably already a numpy array, conversion not necessary
        return list_dicts

    # entering gen: convert _id to sim_id
    for entry in list_dicts:
        if "_id" in entry:
            entry["sim_id"] = entry.pop("_id")

    # first entry is used to determine dtype
    first = list_dicts[0]

    # build a presumptive dtype
    new_dtype_names = _get_new_dtype_fields(first, mapping)
    combinable_names = _get_combinable_multidim_names(first, new_dtype_names)  # [['x0', 'x1'], ['z']]

    if (
        dtype is None
    ):  # rather roundabout. I believe default value gets set upon function instantiation. (default is mutable!)
        dtype = []

    # build dtype of non-mapped fields. appending onto empty dtype
    if not len(dtype):
        dtype = _start_building_dtype(first, new_dtype_names, combinable_names, dtype, mapping)

    # append dtype of mapped float fields
    if len(mapping):
        for name in mapping:
            size = len(mapping[name])
            dtype.append(_decide_dtype(name, 0.0, size))  # float

    out = np.zeros(len(list_dicts), dtype=dtype)

    # starting packing data from list of dicts into array
    for j, input_dict in enumerate(list_dicts):
        for output_name, input_names in zip(new_dtype_names, combinable_names):  # [('x', ['x0', 'x1']), ...]
            if output_name not in mapping:
                out[output_name][j] = _pack_field(input_dict, input_names)
            else:
                out[output_name][j] = _pack_field(input_dict, mapping[output_name])
    return out


def _is_multidim(selection: npt.NDArray) -> bool:
    return hasattr(selection, "__len__") and len(selection) > 1 and not isinstance(selection, str)


def _is_singledim(selection: npt.NDArray) -> bool:
    return hasattr(selection, "__len__") and len(selection) == 1


def np_to_list_dicts(array: npt.NDArray, mapping: dict = {}) -> List[dict]:
    if array is None:
        return None
    out = []

    for row in array:
        new_dict = {}

        for field in row.dtype.names:
            # non-string arrays, lists, etc.

            if field not in list(mapping.keys()):
                if _is_multidim(row[field]):
                    for i, x in enumerate(row[field]):
                        new_dict[field + str(i)] = x

                elif _is_singledim(row[field]):  # single-entry arrays, lists, etc.
                    new_dict[field] = row[field][0]  # will still work on single-char strings

                else:
                    new_dict[field] = row[field]

            # TODO: chase down multivar bug here involving a shape being '()'
            else:  # keys from mapping and array unpacked into corresponding fields in dicts
                assert array.dtype[field].shape[0] == len(mapping[field]), (
                    "dimension mismatch between mapping and array with field " + field
                )

                for i, name in enumerate(mapping[field]):
                    new_dict[name] = row[field][i]

        out.append(new_dict)

    # exiting gen: convert sim_id to _id
    for entry in out:
        if "sim_id" in entry:
            entry["_id"] = entry.pop("sim_id")

    return out
