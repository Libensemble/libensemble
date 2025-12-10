import os
import sys
import subprocess
from pathlib import Path
import shutil

import jinja2
import numpy as np
from heat_flux import heat_flux
from netCDF4 import Dataset

# Optional status codes to display in libE_stats.txt for each gen or sim
from libensemble.message_numbers import TASK_FAILED, WORKER_DONE


def check_DB_and_do_run(x_to_check, sim_specs, libE_info, database_name):
    """Launches the gx MPI app and auto-assigns ranks and GPU resources.

    Assigns one MPI rank to each GPU assigned to the worker.
    """
    calc_status = 0

    DB = []
    match = 0
    if os.path.exists(database_name):
        DB = np.load(database_name,allow_pickle=True)
        for db_entry in DB:
            if np.allclose(db_entry['var_vals'], x_to_check, rtol=1e-12,atol=1e-12):
                print("match",flush=True)
                fout = db_entry['obj_vals']
                string_out = db_entry['string_out']
                match = 1
                calc_status = WORKER_DONE
                break

    if match == 0:
        # Set inputs in input file
        input_file = sim_specs["user"]["input_filename"]
        input_values = {}
        for i, name in enumerate(sim_specs["user"]["input_names"]):
            value = x_to_check[i]
            # if len(H["x"][0]) > 1:
            #     value = H["x"][0][i]
            # else:
            #     value = H["x"][0]
            input_values[name] = value
        with open(input_file, "r") as f:
            template = jinja2.Template(f.read())
        with open(input_file, "w") as f:
            f.write(template.render(input_values))

        nproc = sim_specs["user"]["nproc"]
        nomp = sim_specs["user"]["nomp"]
        numa = sim_specs["user"]["numa"]
        mpinuma = sim_specs["user"]["mpinuma"]
        calc_status = 0

        # Retrieve our MPI Executor
        exctr = libE_info["executor"]
        # env_script_path = "/global/homes/a/arash/bin/cgyro_libe_2"#"/global/u1/a/arash/run_libe_CGYRO_inputs/env_script_in.sh"
        os.environ["OMP_NUM_THREADS"] = "{}".format(nomp)
        # Submit our gx app for execution.

        subprocess.run(["python", "/global/cfs/cdirs/m4493/ebelli/gacode/cgyro/bin/cgyro_parse.py"])

        task = exctr.submit(
            app_name="cgyro",
            app_args="0",
            #procs_per_node=16,  # nl01
            #num_nodes=2, # nl01
            procs_per_node=4,  # reg02
            num_nodes=1, # reg02
            num_gpus=4,
            # auto_assign_gpus=True,
            # match_procs_to_gpus=True,
            # env_script= env_script_path,
            extra_args="--cpu_bind=cores,verbose -n {} -c {}".format(nproc, nomp),
        )

        # Block until the task finishes
        task.wait()

        # Try loading final energy reading, set the sim's status


        string_out = ""
        try:
            # Q=subprocess.run('python heat_flux_cgyro_libE.py', capture_output=True, text=True, shell=True)
            Q = subprocess.run(
                "python /global/u2/j/jmlarson/research/libensemble/examples/run_libe_CGYRO_inputs_0/heat_flux_cgyro_libE.py",
                capture_output=True,
                text=True,
                shell=True,
            )
            freqs = np.loadtxt("out.cgyro.freq")
            fout = freqs[-1]
            with open("out.cgyro.info", 'r') as f:
                lines = f.readlines()
                if lines:
                    string_out = lines[-1].strip()
            calc_status = WORKER_DONE
        except:
            print(f"Failed to open")
            fout = np.nan*np.ones(2)
            calc_status = TASK_FAILED

        to_save = {'obj_vals': fout, 'var_vals': x_to_check, 'string_out': string_out}
        DB = np.append(DB, to_save)
        np.save(database_name,DB)


    # Define our output array, populate with energy reading
    output = np.zeros(1, dtype=sim_specs["out"])
    
    # output["fvec"] = qavg
    # output["f"] = float(qavg[0]) + float(qavg[1])
    # output["fvec"] = fout
    output["f"] = float(fout[1])
    output["convstatement"] = string_out 

    return output, calc_status

def run_CGYRO(H, persis_info, sim_specs, libE_info):

    # database_name = "/global/u2/j/jmlarson/kappa_correction_all.npy"
    database_name = "/global/u2/j/jmlarson/kappa_correction_error.npy"

    output, calc_status = check_DB_and_do_run(np.squeeze(H["x"]), sim_specs, libE_info, database_name) 

    # Return final information to worker, for reporting to manager
    return output, persis_info, calc_status


def run_CGYRO_over_KY(H, persis_info, sim_specs, libE_info):
    database_name = "/global/u2/j/jmlarson/kappa_correction_with_KY_Belli_two.npy"
    output = np.zeros(1, dtype=sim_specs["out"])


    input_file = Path(sim_specs["user"]["input_filename"])
    backup_file = input_file.with_suffix(input_file.suffix + ".safe") if input_file.suffix else Path(str(input_file) + ".safe")

    # Create the backup if it doesn't exist
    if not backup_file.exists():
        shutil.copy2(input_file, backup_file)

    workdir = Path.cwd()

    for i, KY in enumerate(np.arange(0.1,0.65,0.05)):
        x_to_check = np.hstack([np.squeeze(H["x"]), KY])

        # pristine template before running for this KY 
        shutil.copy2(backup_file, input_file)

        # Snapshot files present before the run (after rendering)
        pre_files = set(os.listdir(workdir))

        individual_output, calc_status = check_DB_and_do_run(x_to_check, sim_specs, libE_info, database_name) 
        output["fvec"][0][i] = individual_output["f"]

        # --- Move outputs into a KY-specific directory ---
        run_dir = workdir / f"KY_{KY:.2f}"
        run_dir.mkdir(exist_ok=True)

        # Files that appeared during the run
        post_files = set(os.listdir(workdir))
        new_items = sorted(post_files - pre_files)

        # Exclusions we should never move
        exclusions = {
            backup_file.name,              # keep the .safe backup in place
            run_dir.name,                  # don't move the destination into itself
            Path(database_name).name,      # don't relocate your DB
        }
        # Move newly created files/dirs
        for name in new_items:
            if name in exclusions:
                continue
            src = workdir / name
            try:
                shutil.move(str(src), run_dir / name)
            except Exception:
                # If something resists move (e.g., open/locked), fall back to copy
                if src.is_dir():
                    shutil.copytree(src, run_dir / name, dirs_exist_ok=True)
                    shutil.rmtree(src, ignore_errors=True)
                else:
                    shutil.copy2(src, run_dir / name)
                    try:
                        src.unlink()
                    except Exception:
                        pass

        # Also archive the rendered input file used for this run
        # (It existed before the run snapshot, so it won't be in new_items.)
        try:
            shutil.move(str(input_file), run_dir / input_file.name)
        except Exception:
            shutil.copy2(input_file, run_dir / input_file.name)
            try:
                input_file.unlink()
            except Exception:
                pass

        # (Optional) save a tiny manifest with KY and x_to_check for traceability
        with open(run_dir / "run_manifest.txt", "w") as mf:
            mf.write(f"KY = {KY:.5f}\n")
            mf.write("x_to_check = " + np.array2string(x_to_check, precision=6) + "\n")

        output["f"] = np.max(output["fvec"]) 
        output["convstatement"] = individual_output["convstatement"]

    # Return final information to worker, for reporting to manager
    return output, persis_info, calc_status
