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
from rich.console import Console
from rich.panel import Panel
from rich.align import Align

# Initialize rich console
console = Console(force_terminal=True)

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

platform_mappings = {"Linux":"LIN", "Darwin":"OSX", "Windows":"WIN"}  # Based on TESTSUITE_OS_SKIP
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
    console.print("Cleaning any previous test output...", style="yellow")
    patterns = [
        ".cov_*",
        "ensemble_*",
        "workflow_*",
        "libE_history_at_abort_*.npy",
        "*.out",
        "*.err",
        "*.pickle",
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
        "test_executor_forces_tutorial",
        "test_executor_forces_tutorial_2",

    ]
    dirs_to_clean = UNIT_TEST_DIRS + [REG_TEST_SUBDIR, FUNC_TEST_SUBDIR]
    for dir_path in dirs_to_clean:
        full_path = os.path.join(root_dir, dir_path)
        #import pdb;pdb.set_trace()
        for pattern in patterns:
            for file_path in glob.glob(os.path.join(full_path, pattern)):
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.remove(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    console.print(f"Error removing {file_path}: {e}", style="red")
    console.print("Cleanup completed.\n", style="green")

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
                # SH: Print error output if fail (could show stdout also?)
                # print(stdout)
                print(stderr)
                raise subprocess.CalledProcessError(proc.returncode, cmd)
    else:
        subprocess.run(cmd, cwd=cwd, check=True)

def print_heading(heading, style="bold bright_magenta"):
    """Print a centered panel with the given heading and style."""
    console.print("")
    panel = Panel(Align.center(heading), style=style, expand=True)
    console.print(panel)

def print_summary_line(phrase, style="cyan"):
    """Print a summary line with the specified style."""
    term_width = shutil.get_terminal_size().columns
    line = phrase.center(term_width, "=")
    console.print(f"\n{line}\n", style=style)

def print_test_start(test_num, test_script_name, comm, nprocs):
    """Print the test start message."""
    console.print(
        f"\n---Test {test_num}: {test_script_name} starting with {comm} on {nprocs} processes ", style="cyan"
    )

def print_test_passed(test_num, test_script_name, comm, nprocs, duration, suppress_output):
    """Print the test passed message."""
    if not suppress_output:
        console.print(f" ---Test {test_num}: {test_script_name} using {comm} on {nprocs} processes", end="")
    console.print(f"   ...passed after {duration} seconds", style="green")

def print_test_failed(test_num, test_script_name, comm, nprocs, duration):
    """Print the test failed message."""
    console.print(f" ---Test {test_num}: {test_script_name} using {comm} on {nprocs} processes", end="")
    console.print(f"  ...failed after {duration} seconds", style="red")

def merge_coverage_reports(root_dir):
    """Merge coverage data from multiple tests and generate a report."""
    print_heading("Generating coverage reports")
    tests_dir = os.path.join(root_dir, "libensemble", "tests")
    cov_files = glob.glob(os.path.join(tests_dir, "**", ".cov_*"), recursive=True)

    if cov_files:
        try:
            subprocess.run(["coverage", "combine"] + cov_files, cwd=tests_dir, check=True)
            subprocess.run(["coverage", cov_report_type], cwd=tests_dir, check=True)
            console.print("Coverage reports generated.", style="green")
        except subprocess.CalledProcessError as e:
            console.print("Error generating coverage reports.", style="red")
            sys.exit(e.returncode)
    else:
        console.print("No coverage files found to merge.", style="yellow")

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
        ('# TESTSUITE_OS_SKIP:', 'os_skip', lambda x: x.split()),
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
        if 'Open MPI' in result.stdout or 'OpenRTE' in result.stdout:
            return True
    except Exception:
        pass
    return False

def build_forces(root_dir):
    """Build forces.x using mpicc."""
    console.print("\nBuilding forces.x before running regression tests...", style="yellow")
    forces_app_dir = os.path.join(root_dir, "libensemble/tests/scaling_tests/forces/forces_app")
    subprocess.run(["mpicc", "-O3", "-o", "forces.x", "forces.c", "-lm"], cwd=forces_app_dir, check=True)
    destination_dir = os.path.join(root_dir, "libensemble/tests/forces_app")
    os.makedirs(destination_dir, exist_ok=True)
    shutil.copy(os.path.join(forces_app_dir, "forces.x"), destination_dir)

def skip_test(directives, args, current_os):
    """Skip a test based on directives"""
    if directives['exclude'] or (directives['extra'] and not args.e):
        return True
    if current_os in directives['os_skip']:
        return True
    return False

def skip_config(directives, args, comm):
    """Skip a test configuration based on directives"""
    open_mpi = is_open_mpi()
    mpiexec_flags = args.a if args.a else ''
    if directives['ompi_skip'] and open_mpi and mpiexec_flags == '--oversubscribe' and comm == 'mpi':
        # console.print(f"Skipping test for Open MPI: {test_script_name}")
        return True
    return False

def make_run_line(python_exec, test_script, comm, nprocs, args):
    """Build run line"""
    cmd = python_exec + cov_opts + [test_script]
    if comm == "mpi":
        cmd = ["mpiexec", "-np", str(nprocs)] + (args.a.split() if args.a else []) + cmd
    else:
        cmd += ["--comms", comm, "--nworkers", str(nprocs - 1)]
    return cmd

def process_output(success, test_start, test_num, name, comm, nprocs, suppress):
    """Process output, timing and print"""
    test_end = time.time()
    duration = total_time(test_start, test_end)
    if success:
        print_test_passed(test_num, name, comm, nprocs, duration, suppress)
    else:
        print_test_failed(test_num, name, comm, nprocs, duration)

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
    print_heading(f"Running unit tests (with pytest)")
    for dir_path in UNIT_TEST_DIRS:
        full_path = os.path.join(root_dir, dir_path)
        cov_rep = cov_report_type + ":cov_unit"
        cmd = python_exec + ["-m", "pytest", "--color=yes", "--timeout=120", "--cov", "--cov-report", cov_rep]
        if args.e:
            cmd.append("--runextra")
        if args.s:
            cmd.append("--capture=no")
        run_command(cmd, cwd=full_path)
    console.print("Unit tests completed.\n", style="green")


def run_regression_tests(root_dir, python_exec, args, current_os):
    """Run regression tests."""

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

    print_heading(f"Running regression tests (comms: {', '.join(user_comms_list)})")
    build_forces(root_dir)  # Build forces.x before running tests

    reg_test_list = REG_TEST_LIST
    reg_test_files = []
    for dir_path in test_dirs:
        full_path = os.path.join(root_dir, dir_path)
        reg_test_files.extend(glob.glob(os.path.join(full_path, reg_test_list)))

    reg_test_files = sorted(reg_test_files)
    reg_pass = 0
    reg_fail = 0
    test_num = 0
    start_time = time.time()

    for test_script in reg_test_files:
        test_script_name = os.path.basename(test_script)
        directives = parse_test_directives(test_script)
        if skip_test(directives, args, current_os):
            continue

        comms_list = [comm for comm in directives['comms'] if comm in user_comms_list]
        for comm in comms_list:
            nprocs_list = directives['nprocs']
            for nprocs in nprocs_list:
                if skip_config(directives, args, comm):
                    continue
                test_num += 1
                cmd = make_run_line(python_exec, test_script, comm, nprocs, args)
                test_start = time.time()
                cwd = os.path.dirname(test_script)
                print_test_start(test_num, test_script_name, comm, nprocs)
                try:
                    suppress_output = not args.z
                    # print(f"\nCommand: {' '.join(cmd)}\n")  # For debugging - show run-line
                    run_command(cmd, cwd=cwd, suppress_output=suppress_output)
                    process_output(True, test_start, test_num, test_script_name, comm, nprocs, suppress_output)
                    reg_pass += 1
                except subprocess.CalledProcessError as e:
                    process_output(False, test_start, test_num, test_script_name, comm, nprocs, suppress_output)
                    reg_fail += 1
                    if REG_STOP_ON_FAILURE:
                        sys.exit(e.returncode)
    end_time = time.time()
    total = reg_pass + reg_fail
    summary_style = "green" if reg_fail == 0 else "red"
    prefix = "FAIL" if reg_fail > 0 else "PASS"
    print_summary_line(f" {prefix}: {reg_pass}/{total} regression tests passed in {total_time(start_time, end_time)} seconds ", style=summary_style)
    if reg_fail > 0:
        sys.exit(1)

def main():
    args = parse_arguments()
    root_dir = find_project_root()
    print_heading("************** Running: libEnsemble Test-Suite **************", style="bold bright_yellow")

    cleanup(root_dir)  # Always clean up
    if args.clean:
        sys.exit(0)

    python_exec = ["python"]
    if args.A:
        python_exec += args.A.strip().split()

    global RUN_UNIT_TESTS, RUN_REG_TESTS
    TEST_OPTIONS = [args.u, args.r]
    if any(TEST_OPTIONS):
        RUN_UNIT_TESTS = args.u
        RUN_REG_TESTS = args.r

    base_os = platform.system()
    current_os = platform_mappings.get(base_os)

    # Any introductory info here
    console.print(f"Python executable/options: {' '.join(python_exec)}", style="white")
    console.print(f"OS: {base_os} ({current_os})", style="white")

    if RUN_UNIT_TESTS:
        run_unit_tests(root_dir, python_exec, args)
    if RUN_REG_TESTS:
        run_regression_tests(root_dir, python_exec, args, current_os)

    if COV_REPORT:
        merge_coverage_reports(root_dir)

    console.print("\nAll tests passed.", style="green")

if __name__ == "__main__":
    main()
