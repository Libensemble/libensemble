def sim_inputs(r):
    def wrapper(f):
        f.inputs = r
        return f
    return wrapper

def sim_outputs(r):
    def wrapper(f):
        f.outputs = r
        return f
    return wrapper