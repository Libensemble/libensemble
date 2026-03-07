"""
Misc internal functions
"""

from itertools import chain, groupby
from operator import itemgetter
from typing import List

import numpy as np
import numpy.typing as npt


def extract_H_ranges(Work: dict) -> str:
    """Convert received H_rows into ranges for labeling"""
    work_H_rows = Work["libE_info"]["H_rows"]
    if len(work_H_rows) == 1:
        return str(work_H_rows[0])
    else:
        # From https://stackoverflow.com/a/30336492
        ranges = []
        for diff, group_iter in groupby(enumerate(work_H_rows.tolist()), lambda x: x[0] - x[1]):
            group = list(map(itemgetter(1), group_iter))
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
    return specs.model_dump(**kwargs)


def specs_checker_getattr(obj, key, default=None):
    try:
        return getattr(obj, key)
    except AttributeError:
        return default


def specs_checker_setattr(obj, key, value):
    obj.__dict__[key] = value


def _combine_names(names: list) -> list:
    """Return unique field names without auto-combining"""
    return list(dict.fromkeys(names))  # preserves order, removes duplicates


def _get_new_dtype_fields(first: dict, mapping: dict = {}) -> list:
    """build list of fields that will be in the output numpy array"""
    new_dtype_names = _combine_names([i for i in first.keys()])  # -> ['x', 'y']
    fields_to_convert = list(  # combining all mapping lists
        chain.from_iterable(list(mapping.values()))
    )  # fields like ["beam_length", "beam_width"] that will become "x"
    new_dtype_names = [i for i in new_dtype_names if i not in fields_to_convert]
    return new_dtype_names


def _get_combinable_multidim_names(first: dict, new_dtype_names: list) -> list:
    """Return each field name as a single-element list without auto-grouping"""
    return [[name] for name in new_dtype_names]


def _decide_dtype(name: str, entry, size: int) -> tuple:
    """decide dtype of field, and size if needed"""
    if isinstance(entry, str):  # use numpy style for string type
        output_type = "U" + str(len(entry) + 1)
    else:
        output_type = type(entry)  # use default "python" type
    if name == "sim_id":
        output_type = int
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
    """Convert list of dicts to numpy structured array"""
    if list_dicts is None:
        return None

    if not isinstance(list_dicts, list):
        return list_dicts

    if not list_dicts:
        return np.array([], dtype=dtype if dtype else [])

    # first entry is used to determine dtype
    first = list_dicts[0]

    # build a presumptive dtype
    new_dtype_names = _get_new_dtype_fields(first, mapping)
    combinable_names = _get_combinable_multidim_names(first, new_dtype_names)  # [['x0', 'x1'], ['z']]

    if dtype is None:  # Default value gets set upon function instantiation (default is mutable).
        dtype = []

    # build dtype of non-mapped fields. appending onto empty dtype
    if not len(dtype):
        dtype = _start_building_dtype(first, new_dtype_names, combinable_names, dtype, mapping)

    # append dtype of mapped float fields
    if len(mapping):
        existing_names = [f[0] for f in dtype]
        for name in mapping:
            # If the field is already in the dtype, skip it. *And* the field is present in the input data
            if name not in existing_names and all(src in first for src in mapping[name]):
                size = len(mapping[name])
                dtype.append(_decide_dtype(name, 0.0, size))  # default to float
                new_dtype_names.append(name)
                combinable_names.append(mapping[name])

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
    return (hasattr(selection, "__len__") and len(selection) == 1) or selection.shape == ()


def unmap_numpy_array(array: npt.NDArray, mapping: dict = {}) -> npt.NDArray:
    """Convert numpy array with mapped fields back to individual scalar fields.
    Parameters
    ----------
    array : npt.NDArray
        Input array with mapped fields like x = [x0, x1, x2]
    mapping : dict
        Mapping from field names to variable names
    Returns
    -------
    npt.NDArray
        Array with unmapped fields like x0, x1, x2 as individual scalars
    """
    if not mapping or array is None:
        return array
    # Create new dtype with unmapped fields
    new_fields = []
    for field in array.dtype.names:
        if field in mapping:
            for var_name in mapping[field]:
                new_fields.append((var_name, array[field].dtype.type))
        else:
            # Preserve the original field structure including per-row shape
            field_dtype = array.dtype[field]
            new_fields.append((field, field_dtype))
    unmapped_array = np.zeros(len(array), dtype=new_fields)
    for field in array.dtype.names:
        if field in mapping:
            # Unmap array fields
            if len(array[field].shape) == 1:
                # Scalar field mapped to single variable
                unmapped_array[mapping[field][0]] = array[field]
            else:
                # Multi-dimensional field
                for i, var_name in enumerate(mapping[field]):
                    unmapped_array[var_name] = array[field][:, i]
        else:
            # Copy non-mapped fields
            unmapped_array[field] = array[field]
    return unmapped_array


def map_numpy_array(array: npt.NDArray, mapping: dict = {}) -> npt.NDArray:
    """Convert numpy array with individual scalar fields to mapped fields.
    Parameters
    ----------
    array : npt.NDArray
        Input array with unmapped fields like x0, x1, x2
    mapping : dict
        Mapping from field names to variable names
    Returns
    -------
    npt.NDArray
        Array with mapped fields like x = [x0, x1, x2]
    """
    if not mapping or array is None:
        return array

    # Create new dtype with mapped fields
    new_fields = []

    # Track fields processed by mapping to avoid duplication
    mapped_source_fields = set()
    for key, val_list in mapping.items():
        mapped_source_fields.update(val_list)

    # First add mapped fields from the mapping definition
    for mapped_name, val_list in mapping.items():
        first_var = val_list[0]
        # We assume all components have the same type, take from first
        base_type = array.dtype[first_var]
        size = len(val_list)
        if size > 1:
            new_fields.append((mapped_name, base_type, (size,)))
        else:
            new_fields.append((mapped_name, base_type))

    # Then add any fields from the source array that were NOT part of a mapping
    for field in array.dtype.names:
        if field not in mapped_source_fields:
            new_fields.append((field, array.dtype[field]))

    # remove duplicates from new_fields
    new_fields = list(dict.fromkeys(new_fields))

    # Create the new array
    mapped_array = np.zeros(len(array), dtype=new_fields)

    # Fill the new array
    for field in mapped_array.dtype.names:
        # Mapped field: stack the source columns
        val_list = mapping[field]
        if len(val_list) == 1:
            mapped_array[field] = array[val_list[0]]
        else:
            # Stack columns horizontally for each row
            # We need to extract each column, then stack them along axis 1
            cols = [array[val] for val in val_list]
            mapped_array[field] = np.stack(cols, axis=1)

    return mapped_array


def np_to_list_dicts(array: npt.NDArray, mapping: dict = {}) -> List[dict]:
    """Convert numpy structured array to list of dicts"""
    out = []

    for row in array:
        new_dict = {}

        for field in row.dtype.names:
            if field not in list(mapping.keys()):
                # Unmapped fields: copy directly (no auto-unpacking)
                new_dict[field] = row[field]
            else:  # keys from mapping and array unpacked into corresponding fields in dicts
                field_shape = array.dtype[field].shape[0] if len(array.dtype[field].shape) > 0 else 1
                assert field_shape == len(mapping[field]), (
                    "dimension mismatch between mapping and array with field " + field
                )

                for i, name in enumerate(mapping[field]):
                    if _is_multidim(row[field]):
                        new_dict[name] = row[field][i]
                    elif _is_singledim(row[field]):
                        new_dict[name] = row[field]

        out.append(new_dict)

    # Remove _id from entries where it's -1 (unset)
    for entry in out:
        if entry.get("_id") == -1:
            entry.pop("_id")

    return out
