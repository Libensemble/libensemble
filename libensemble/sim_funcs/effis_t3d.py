#!/usr/bin/env python3

import os
import re

import effis.composition
import numpy as np
import t3d.trinity_lib


def effis_t3d(H_in, _, sim_specs, libE_info):

    # COMPILATION STEP

    t3dir = os.path.dirname(t3d.trinity_lib.__file__)
    setupdir = os.path.join(os.path.dirname(t3dir), "tests", "regression")
    datadir = os.path.join(os.path.dirname(t3dir), "tests", "data")
    configname = "test-w7x-relu"

    x0 = H_in["x"][0][0]
    x1 = H_in["x"][0][1]

    exctr = libE_info["executor"]

    runner = effis.composition.Workflow.DetectRunnerInfo()

    MyWorkflow = effis.composition.Workflow(
        Runner=runner,
        Directory="output",
    )
    MyWorkflow.GroupMax["t3d"] = 2

    Simulation = MyWorkflow.Application(cmd="t3d", Name="T3D-1", Ranks=1, Group="t3d")
    Simulation.CommandLineArguments += [
        "{0}.in".format(configname),
        "--log",
        "{0}.out".format(configname),
    ]

    Simulation.Input += effis.composition.Input(os.path.join(setupdir, "{0}.in".format(configname)))
    Simulation.Input += effis.composition.Input(os.path.join(datadir, "wout_w7x.nc"), link=True)

    plot = MyWorkflow.Application(
        cmd="t3d-plot",
        Name="plot-1",
        Runner=None,
    )
    plot.CommandLineArguments += [
        os.path.relpath(
            os.path.join(Simulation.Directory, "{0}.bp".format(configname)),
            start=plot.Directory,
        ),
        "--grid",
        "--savefig",
        "-p",
        "density",
        "temperature",
        "pressure",
        "heat_flux",
        "particle_flux",
        "flux",
    ]
    plot.DependsOn += Simulation

    MyWorkflow.Create()

    for Simulation in MyWorkflow.Applications:

        if Simulation.Name.find("plot") != -1:
            continue

        # Rewrite a few things in the config file
        with open(os.path.join(Simulation.Directory, "{0}.in".format(configname)), "r") as infile:
            config = infile.read()
        config = re.compile(r"geo_file\s*=\s*.*", re.MULTILINE).sub('geo_file = "wout_w7x.nc"', config)

        core = 0.35 + x0 * 0.01
        edge = 0.29 + x1 * 0.01
        config = re.compile(r"density = {core = 0.35, edge = 0.29, alpha=1, evolve = false}", re.MULTILINE,).sub(
            "density = {{core = {0}, edge = {1}, alpha=1, evolve = false}}".format(core, edge),
            config,
        )

        with open(os.path.join(Simulation.Directory, "{0}.in".format(configname)), "w") as outfile:
            outfile.write(config)

    MyWorkflow.PickleWrite()  # The workflow usually only pickles *when the script exits* ???

    # END COMPILATION STEP

    # START SIM STEP
    app_args = "output"
    task = exctr.submit(app_name="effis", app_args=app_args, wait_on_start=3)
    calc_status = exctr.polling_loop(task, timeout=8, delay=0.3)

    output_data = np.load("./output/T3D-1/test-w7x-relu.log.npy", allow_pickle=True).flatten()[0]

    H_out = np.zeros(1, dtype=sim_specs["out"])
    H_out["Wtot_MJ"] = output_data["Wtot_MJ"][-1]

    return H_out, _, calc_status
