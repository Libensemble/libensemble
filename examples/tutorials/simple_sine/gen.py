import numpy as np


def gen_random_sample(Input, persis_info, gen_specs):
    # Pull out user parameters
    user_specs = gen_specs["user"]

    # Get lower and upper bounds
    lower = user_specs["lower"]
    upper = user_specs["upper"]

    # Determine how many values to generate
    num = len(lower)
    batch_size = user_specs["gen_batch_size"]

    # Create empty array of "batch_size" zeros. Array dtype should match "out" fields
    Output = np.zeros(batch_size, dtype=gen_specs["out"])

    # Set the "x" output field to contain random numbers, using random stream
    Output["x"] = persis_info["rand_stream"].uniform(lower, upper, (batch_size, num))

    # Send back our output and persis_info
    return Output, persis_info
