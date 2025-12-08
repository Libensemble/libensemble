"""
Wrapper for simulation functions in the gest-api format.

Gest-api functions take an input_dict (single point as dictionary) with
VOCS variables and constants, and return a dict with VOCS objectives,
observables, and constraints.
"""

import numpy as np

__all__ = ["gest_api_sim"]


def gest_api_sim(H, persis_info, sim_specs, libE_info):
    """
    LibEnsemble sim_f wrapper for gest-api format simulation functions.

    Converts between libEnsemble's numpy structured array format and
    gest-api's dictionary format for individual points.

    Parameters
    ----------
    H : numpy structured array
        Input points from libEnsemble containing VOCS variables and constants
    persis_info : dict
        Persistent information dictionary
    sim_specs : dict
        Simulation specifications. Must contain:
        - "vocs": VOCS object defining variables, constants, objectives, etc.
        - "simulator": The gest-api function
    libE_info : dict
        LibEnsemble information dictionary

    Returns
    -------
    H_o : numpy structured array
        Output array with VOCS objectives, observables, and constraints
    persis_info : dict
        Updated persistent information

    Notes
    -----
    The gest-api simulator function should have signature:
        def simulator(input_dict: dict, **kwargs) -> dict

    Where input_dict contains VOCS variables and constants,
    and the return dict contains VOCS objectives, observables, and constraints.

    If the simulator function accepts ``libE_info``, it will be passed. This
    allows simulators to access libEnsemble information such as the executor.
    """

    simulator = sim_specs["simulator"]
    vocs = sim_specs["vocs"]
    user_specs = sim_specs.get("user", {})

    batch = len(H)
    H_o = np.zeros(batch, dtype=sim_specs["out"])

    # Helper to get fields from VOCS (handles both object and dict)
    def get_vocs_fields(vocs, attr_names):
        fields = []
        is_object = hasattr(vocs, attr_names[0])
        for attr in attr_names:
            obj = getattr(vocs, attr, None) if is_object else vocs.get(attr)
            if obj:
                fields.extend(list(obj.keys()))
        return fields

    # Get input fields (variables + constants) and output fields (objectives + observables + constraints)
    input_fields = get_vocs_fields(vocs, ["variables", "constants"])
    output_fields = get_vocs_fields(vocs, ["objectives", "observables", "constraints"])

    # Process each point in the batch
    for i in range(batch):
        # Build input_dict from H for this point
        input_dict = {}
        for field in input_fields:
            input_dict[field] = H[field][i]

        # Try to pass libE_info, fall back if function doesn't accept it
        try:
            output_dict = simulator(input_dict, libE_info=libE_info, **user_specs)
        except TypeError:
            # Function doesn't accept libE_info, call without it
            output_dict = simulator(input_dict, **user_specs)

        # Extract outputs from the returned dict
        for field in output_fields:
            if field in output_dict:
                H_o[field][i] = output_dict[field]

    return H_o, persis_info
