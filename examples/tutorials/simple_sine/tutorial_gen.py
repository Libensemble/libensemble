import numpy as np


def gen_random_sample(H, persis_info, gen_specs, _):
    # underscore parameter for internal/testing arguments

    # Pull out user parameters to perform calculations
    user_specs = gen_specs["user"]

    # Get lower and upper bounds from gen_specs
    lower = user_specs["lower"]
    upper = user_specs["upper"]

    # Determine how many values to generate
    num = len(lower)
    batch_size = user_specs["gen_batch_size"]

    # Create array of 'batch_size' zeros
    out = np.zeros(batch_size, dtype=gen_specs["out"])

    # Replace those zeros with the random numbers
    out["x"] = persis_info["rand_stream"].uniform(lower, upper, (batch_size, num))

    # Send back our output and persis_info
    return out, persis_info
