import pycutest, os

def print_metadata():
    # Find unconstrained, variable-dimension problems
    probs = pycutest.find_problems(objective='LQS', constraints='U', userN=True, regular=True)
    probs = sorted(probs)
    # print('List of {} Possible problems: {}'.format(len(probs), probs))

    for prob in probs:
        print('Name={}'.format(prob))
        # print(pycutest.problem_properties(prob))
        pycutest.print_available_sif_params(prob)
        print('End=')

print_metadata()
