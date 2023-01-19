import os
import ast
import subprocess

# SH TODO while testing - REMOVE
from libensemble.tools import eprint

# SH TODO - Remove eprints
# import sys
# def eprint(*args, **kwargs):
#    """Prints a user message to standard error"""
#    print(*args, file=sys.stderr, **kwargs)


def pynvml():
    """Detect GPU from pynvml or return None"""
    try:
        import pynvml

        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        eprint("gpu count from pynvml", gpu_count)
    except Exception:
        eprint("pynvml (optional) not found or failed")
        return None
    return gpu_count


def nvidia_smi():
    """Detect GPU from nvidia-smi or return None"""
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"])
        gpu_count = len(output.decode().split())
        eprint("gpu count from nvidia-smi", gpu_count)
    except Exception:
        eprint("nvidia-smi (optional) not found or failed")
        return None
    return gpu_count


def pyadl():
    """Detect GPU from pyadl or return None"""
    try:
        from pyadl import ADLManager

        devices = ADLManager.getInstance().getDevices()
        gpu_count = len(devices)
        eprint("gpu count from pyadl", gpu_count)
    except Exception:
        eprint("pyadl (optional) not found or failed")
        return None
    return gpu_count


def rocm_smi():
    """Detect GPU from rocm-smi or return None"""
    try:
        output = subprocess.check_output(["rocm-smi", "-i", "--json"])
        gpu_count = len(ast.literal_eval(output.decode()))
        eprint("gpu count from rocm-smi", gpu_count)
    except Exception:
        eprint("rocm-smi (optional) not found or failed")
        return None
    return gpu_count


def zeinfo():
    """Detect GPU from zeinfo or return None"""
    try:
        ps = subprocess.Popen(('zeinfo'), stderr=subprocess.PIPE)
        output = subprocess.check_output(('grep', 'Number of devices'), stdin=ps.stderr)
        gpu_count = int(output.decode().split()[3])
        eprint("gpu count from zeinfo", gpu_count)
    except Exception:
        eprint("zeinfo (optional) not found or failed")
        return None
    return gpu_count


METHODS = {
    "pynvml": pynvml,
    "nvidia_smi": nvidia_smi,
    "pyadl": pyadl,
    "rocm_smi": rocm_smi,
    "zeinfo": zeinfo,
}


def get_num_gpus(testall=False):
    """Return number of GPUs on node if can detect - else None"""

    # Default zero or None
    gpu_count = None

    for method in METHODS:
        gpu_count = METHODS[method]()
        if isinstance(gpu_count, int) and not testall:
            return gpu_count

    return None


def get_gpus_from_env(env_resources=None):
    """Returns gpus per node by querying environment or None"""

    if not env_resources:
        return None

    if env_resources.scheduler == "Slurm":
        gpu_count = os.getenv("SLURM_GPUS_ON_NODE")
        eprint("gpu count from SLURM_GPUS_ON_NODE", gpu_count)
        # return os.getenv("SLURM_GPUS_ON_NODE")
        if gpu_count is not None:
            return int(gpu_count)

    return None


# for testing
if __name__ == "__main__":
    get_num_gpus(testall=True)
