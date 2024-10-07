#!/usr/bin/env python3

import argparse
import os
import sys
import subprocess
import shutil
import glob
import time
import platform
from pathlib import Path

# Import colorama for colored output
from colorama import init, Fore

sys.stdout.isatty = lambda: True

# Initialize colorama
init(autoreset=True)
#init(autoreset=True, convert=True, strip=False)

# -----------------------------------------------------------------------------------------
# Configuration

# Default options
RUN_UNIT_TESTS = True
RUN_REG_TESTS = True
COV_REPORT = True

# Regression test options
REG_TEST_LIST = "test_*.py"
REG_TEST_OUTPUT_EXT = "std.out"
REG_STOP_ON_FAILURE = False

# Test Directories - all relative to project root dir
CODE_DIR = "libensemble"
LIBE_SRC_DIR = CODE_DIR
TESTING_DIR = os.path.join(CODE_DIR, "tests")
UNIT_TEST_SUBDIRS = [
    "unit_tests",
    "unit_tests_mpi_import",
    "unit_tests_nompi",
    "unit_tests_logger",
]
UNIT_TEST_DIRS = [os.path.join(TESTING_DIR, subdir) for subdir in UNIT_TEST_SUBDIRS]
REG_TEST_SUBDIR = os.path.join(TESTING_DIR, "regression_tests")
FUNC_TEST_SUBDIR = os.path.join(TESTING_DIR, "functionality_tests")

# Coverage merge and report dir
COV_MERGE_DIR = TESTING_DIR

cov_opts = ["-m", "coverage", "run", "--parallel-mode", "--concurrency=multiprocessing,thread"]
cov_report_type = "xml"  # e.g., html, xml

# -----------------------------------------------------------------------------------------
# Environment Variables

# Set environment variables as needed
os.environ["OMPI_SKIP_MPICXX"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# -----------------------------------------------------------------------------------------
# Helper Functions

def find_project_root():
    """Find the project root directory."""
    root_dir = Path(__file__).resolve().parents[2]
    return str(root_dir)

def cleanup(root_dir):
    """Cleanup test run directories."""
    print("Cleaning up test output...")
    patterns = [
        ".cov_merge_out*",
        "ensemble_*",
        "workflow_*",
        "libE_history_at_abort_*.npy",
        "*.out",
        "*.err",
        "*.pickle",
        ".cov_unit_out*",
        "simdir/*.x",
        "libe_task_*.out",
        "*libE_stats.txt",
        "my_machinefile",
        "libe_stat_files",
        "ensemble.log",
        "H_test.npy",
        "workflow_intermediate*",
        f"*.{REG_TEST_OUTPUT_EXT}",
        "*.npy",
        "*active_runs.txt",
        "outfile*.txt",
        "machinefile*",
        "my_simtask.x",
        "sim_*",
        "gen_*",
        "nodelist_*",
        "x_*.txt",
        "y_*.txt",
        "opt_*.txt_flag",
        "*.png",
    ]
    dirs_to_clean = UNIT_TEST_DIRS + [REG_TEST_SUBDIR, FUNC_TEST_SUBDIR]
    for dir_path in dirs_to_clean:
        full_path = os.path.join(root_dir, dir_path)
        for pattern in patterns:
            for file_path in glob.glob(os.path.join(full_path, pattern)):
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")
    print("Cleanup completed.")

def total_time(start, end):
    """Return a time difference."""
    return round(end - start, 2)

def run_command(cmd, cwd=None, suppress_output=False):
    """Run a shell command and display its output."""
    if suppress_output:
        with subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        ) as proc:
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, cmd)
    else:
        print(f"\n{cmd=}\n")
        subprocess.run(cmd, cwd=cwd, check=True)

def print_summary_line(phrase):
    """Print a summary line like pytest."""
    term_width = shutil.get_terminal_size().columns
    line = phrase.center(term_width, "=")
    print(line)

def print_test_start(test_num, test_script_name, comm, nprocs):
    """Print the test start message."""
    print(
        Fore.CYAN
        + f"\n ---Test {test_num}: {test_script_name} starting with {comm} on {nprocs} processes "
    )

def print_test_passed(test_num, test_script_name, comm, nprocs, duration):
    """Print the test passed message."""
    print(
        f" ---Test {test_num}: {test_script_name} using {comm} on {nprocs} processes"
        + Fore.GREEN
        + f"  ...passed after {duration} seconds"
    )

def print_test_failed(test_num, test_script_name, comm, nprocs, duration):
    """Print the test failed message."""
    print(
        f" ---Test {test_num}: {test_script_name} using {comm} on {nprocs} processes"
        + Fore.RED
        + f"  ...failed after {duration} seconds"
    )

def merge_coverage_reports(root_dir):
    """Merge coverage data from multiple tests and generate a report."""
    print(Fore.CYAN + "\nGenerating coverage reports...")

    # Append libensemble/tests to root_dir in both glob and subprocess
    tests_dir = os.path.join(root_dir, "libensemble", "tests")
    cov_files = glob.glob(os.path.join(tests_dir, "**", ".cov_*"), recursive=True)

    if cov_files:
        try:
            subprocess.run(["coverage", "combine"] + cov_files, cwd=tests_dir, check=True)
            subprocess.run(["coverage", cov_report_type], cwd=tests_dir, check=True)
            print(Fore.GREEN + "Coverage reports generated.")
        except subprocess.CalledProcessError as e:
            print(Fore.RED + "Error generating coverage reports.")
            sys.exit(e.returncode)
    else:
        print(Fore.YELLOW + "No coverage files found to merge.")

def parse_test_directives(test_script):
    """Parse test suite directives from the test script."""
    directives = {
        'comms': ['mpi', 'local', 'tcp'],
        'nprocs': [4],
        'extra': False,
        'exclude': False,
        'os_skip': [],
        'ompi_skip': False,
    }

    directive_patterns = [
        ('# TESTSUITE_COMMS:', 'comms', lambda x: x.split()),
        ('# TESTSUITE_NPROCS:', 'nprocs', lambda x: [int(n) for n in x.split()]),
        ('# TESTSUITE_EXTRA:', 'extra', lambda x: x.lower() == 'true'),
        ('# TESTSUITE_EXCLUDE:', 'exclude', lambda x: x.lower() == 'true'),
        ('# TESTSUITE_OS_SKIP:', 'os_skip', lambda x: x.lower().split()),
        ('# TESTSUITE_OMPI_SKIP:', 'ompi_skip', lambda x: x.lower() == 'true'),
    ]

    with open(test_script, 'r') as f:
        for line in f:
            for pattern, key, parse_func in directive_patterns:
                if line.startswith(pattern):
                    value = line.split(':', 1)[1].strip()
                    directives[key] = parse_func(value)
                    break
    return directives

def is_open_mpi():
    """Check if Open MPI is being used."""
    try:
        result = subprocess.run(['mpiexec', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if 'Open MPI' in result.stdout:
            return True
    except Exception:
        pass
    return False

# -----------------------------------------------------------------------------------------
# Main Functions

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="LibEnsemble Test Runner")
    parser.add_argument("-c", "--clean", action="store_true", help="Clean up test directories and exit")
    parser.add_argument("-s", action="store_true", help="Print stdout and stderr during pytest (unit tests)")
    parser.add_argument("-z", action="store_true", help="Print stdout and stderr during regression tests")
    parser.add_argument("-u", action="store_true", help="Run only the unit tests")
    parser.add_argument("-r", action="store_true", help="Run only the regression tests")
    parser.add_argument("-m", action="store_true", help="Run the regression tests using MPI comms")
    parser.add_argument("-l", action="store_true", help="Run the regression tests using Local comms")
    parser.add_argument("-t", action="store_true", help="Run the regression tests using TCP comms")
    parser.add_argument("-e", action="store_true", help="Run extra unit and regression tests")
    parser.add_argument("-A", metavar="<string>", help="Supply arguments to python")
    parser.add_argument("-a", metavar="<string>", help="Supply a string of args to add to mpiexec line")
    args = parser.parse_args()
    return args

def run_unit_tests(root_dir, python_exec, args):
    """Run unit tests."""
    print(Fore.CYAN + "\nRunning unit tests...")
    for dir_path in UNIT_TEST_DIRS:
        full_path = os.path.join(root_dir, dir_path)
        cov_rep = cov_report_type + ":cov_unit"
        cmd = python_exec + ["-m", "pytest", "--timeout=120", "--cov", "--cov-report", cov_rep]
        if args.e:
            cmd.append("--runextra")
        if args.s:
            cmd.append("--capture=no")
        run_command(cmd, cwd=full_path)
    print(Fore.GREEN + "Unit tests completed.")

def build_forces(root_dir):
    """Build forces.x using mpicc."""
    forces_app_dir = os.path.join(root_dir, "libensemble/tests/scaling_tests/forces/forces_app")
    subprocess.run(["mpicc", "-O3", "-o", "forces.x", "forces.c", "-lm"], cwd=forces_app_dir, check=True)
    destination_dir = os.path.join(root_dir, "libensemble/tests/forces_app")
    os.makedirs(destination_dir, exist_ok=True)
    shutil.copy(os.path.join(forces_app_dir, "forces.x"), destination_dir)

def run_regression_tests(root_dir, python_exec, args):
    """Run regression tests."""
    print(Fore.CYAN + "\nBuilding forces.x before running regression tests...")
    print(f"{os.getcwd()}")
    build_forces(root_dir)  # Build forces.x before running tests

    print(Fore.CYAN + "\nRunning regression tests...")
    test_dirs = [REG_TEST_SUBDIR, FUNC_TEST_SUBDIR]
    user_comms_list = []
    if args.m:
        user_comms_list.append("mpi")
    if args.l:
        user_comms_list.append("local")
    if args.t:
        user_comms_list.append("tcp")
    if not user_comms_list:
        user_comms_list = ["mpi", "local", "tcp"]

    reg_test_list = REG_TEST_LIST
    reg_test_files = []
    for dir_path in test_dirs:
        full_path = os.path.join(root_dir, dir_path)
        reg_test_files.extend(glob.glob(os.path.join(full_path, reg_test_list)))

    # Sort the test files alphabetically
    reg_test_files = sorted(reg_test_files)

    reg_pass = 0
    reg_fail = 0
    start_time = time.time()
    test_num = 0
    current_os = platform.system().lower()  # Get current OS
    open_mpi = is_open_mpi()
    mpiexec_flags = args.a if args.a else ''

    for test_script in reg_test_files:
        test_script_name = os.path.basename(test_script)
        directives = parse_test_directives(test_script)

        if directives['exclude'] or (directives['extra'] and not args.e):
            continue
        if current_os in directives['os_skip']:
            continue  # Skip this test on the current OS

        comms_list = [comm for comm in directives['comms'] if comm in user_comms_list]

        for comm in comms_list:
            nprocs_list = directives['nprocs']
            for nprocs in nprocs_list:
                test_num += 1

                # Check for Open MPI skip condition
                if directives['ompi_skip'] and open_mpi and mpiexec_flags == '--oversubscribe' and comm == 'mpi':
                    print(f"Skipping test number {test_num} for Open MPI: {test_script_name}")
                    continue

                cmd = python_exec + cov_opts + [test_script]
                if comm == "mpi":
                    cmd = ["mpiexec", "-np", str(nprocs)] + (args.a.split() if args.a else []) + cmd
                else:
                    cmd += ["--comms", comm, "--nworkers", str(nprocs - 1)]

                test_start = time.time()
                cwd = os.path.dirname(test_script)
                print_test_start(test_num, test_script_name, comm, nprocs)
                try:
                    suppress_output = not args.z
                    run_command(cmd, cwd=cwd, suppress_output=suppress_output)
                    test_end = time.time()
                    duration = total_time(test_start, test_end)
                    print_test_passed(test_num, test_script_name, comm, nprocs, duration)
                    reg_pass += 1
                except subprocess.CalledProcessError as e:
                    test_end = time.time()
                    duration = total_time(test_start, test_end)
                    print_test_failed(test_num, test_script_name, comm, nprocs, duration)
                    reg_fail += 1
                    if REG_STOP_ON_FAILURE:
                        sys.exit(e.returncode)
    end_time = time.time()
    total = reg_pass + reg_fail
    if reg_fail == 0:
        summary_color = Fore.GREEN
    else:
        summary_color = Fore.RED
    print_summary_line(
        f"{summary_color}{reg_pass}/{total} regression tests passed in {total_time(start_time, end_time)} seconds"
    )
    if reg_fail > 0:
        sys.exit(1)


def main():
    args = parse_arguments()
    root_dir = find_project_root()

    if args.clean:
        cleanup(root_dir)  #SH think should run clean by default - but exit if -c. Check safety of clean.
        sys.exit(0)

    # Set Python executable
    python_exec = ["python"]
    if args.A:
        python_exec += args.A.strip().split()  #SH do i want this for unit tests

    # Set options based on arguments
    global RUN_UNIT_TESTS, RUN_REG_TESTS
    TEST_OPTIONS = [args.u, args.r]
    if any(TEST_OPTIONS):
        RUN_UNIT_TESTS = args.u
        RUN_REG_TESTS = args.r

    print(Fore.CYAN + f"Python executable: {' '.join(python_exec)}")
    if RUN_UNIT_TESTS:
        run_unit_tests(root_dir, python_exec, args)
    if RUN_REG_TESTS:
        run_regression_tests(root_dir, python_exec, args)

    if COV_REPORT:
        merge_coverage_reports(root_dir)

    print(Fore.GREEN + "\nAll tests passed.")

if __name__ == "__main__":
    main()
