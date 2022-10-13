from libensemble.resources import mpi_resources
from libensemble.executors.executor import jassert
import argparse
import logging

logger = logging.getLogger(__name__)
# To change logging level for just this module
# logger.setLevel(logging.DEBUG)


class MPIRunner:
    @staticmethod
    def get_runner(mpi_runner_type, runner_name=None):

        mpi_runners = {
            "mpich": MPICH_MPIRunner,
            "openmpi": OPENMPI_MPIRunner,
            "aprun": APRUN_MPIRunner,
            "srun": SRUN_MPIRunner,
            "jsrun": JSRUN_MPIRunner,
            "msmpi": MSMPI_MPIRunner,
            "custom": MPIRunner,
        }
        mpi_runner = mpi_runners[mpi_runner_type]
        if runner_name is not None:
            runner = mpi_runner(runner_name)
        else:
            runner = mpi_runner()
        return runner

    def __init__(self, run_command="mpiexec"):
        self.run_command = run_command
        self.mpi_command = [self.run_command, "{extra_args}"]
        self.subgroup_launch = False
        self.mfile_support = False
        self.arg_nprocs = ("--LIBE_NPROCS_ARG_EMPTY",)
        self.arg_nnodes = ("--LIBE_NNODES_ARG_EMPTY",)
        self.arg_ppn = ("--LIBE_PPN_ARG_EMPTY",)

    def _get_parser(self, p_args, nprocs, nnodes, ppn):
        parser = argparse.ArgumentParser(description="Parse extra_args", allow_abbrev=False)
        parser.add_argument(*nprocs, type=int, dest="num_procs", default=None)
        parser.add_argument(*nnodes, type=int, dest="num_nodes", default=None)
        parser.add_argument(*ppn, type=int, dest="procs_per_node", default=None)
        args, _ = parser.parse_known_args(p_args)
        return args

    def _parse_extra_args(self, num_procs, num_nodes, procs_per_node, hyperthreads, extra_args):

        splt_extra_args = extra_args.split()
        p_args = self._get_parser(splt_extra_args, self.arg_nprocs, self.arg_nnodes, self.arg_ppn)

        # Only fill from extra_args if not set by portable options
        if num_procs is None:
            num_procs = p_args.num_procs
        if num_nodes is None:
            num_nodes = p_args.num_nodes
        if procs_per_node is None:
            procs_per_node = p_args.procs_per_node

        extra_args = " ".join(splt_extra_args)
        return num_procs, num_nodes, procs_per_node, p_args

    def _rm_replicated_args(self, num_procs, num_nodes, procs_per_node, p_args):
        if p_args.num_procs is not None:
            num_procs = None
        if p_args.num_nodes is not None:
            num_nodes = None
        if p_args.procs_per_node is not None:
            procs_per_node = None
        return num_procs, num_nodes, procs_per_node

    def express_spec(
        self, task, num_procs, num_nodes, procs_per_node, machinefile, hyperthreads, extra_args, resources, workerID
    ):

        hostlist = None
        machinefile = None
        # Always use host lists (unless uneven mapping)
        hostlist = mpi_resources.get_hostlist(resources, num_nodes)
        return hostlist, machinefile

    def get_mpi_specs(
        self, task, num_procs, num_nodes, procs_per_node, machinefile, hyperthreads, extra_args, resources, workerID
    ):
        "Form the mpi_specs dictionary."

        # Return auto_resource variables inc. extra_args additions
        if extra_args:
            num_procs, num_nodes, procs_per_node, p_args = self._parse_extra_args(
                num_procs, num_nodes, procs_per_node, hyperthreads, extra_args=extra_args
            )

        hostlist = None
        if machinefile and not self.mfile_support:
            logger.warning(f"User machinefile ignored - not supported by {self.run_command}")
            machinefile = None

        if machinefile is None and resources is not None:
            num_procs, num_nodes, procs_per_node = mpi_resources.get_resources(
                resources, num_procs, num_nodes, procs_per_node, hyperthreads
            )
            hostlist, machinefile = self.express_spec(
                task, num_procs, num_nodes, procs_per_node, machinefile, hyperthreads, extra_args, resources, workerID
            )
        else:
            num_procs, num_nodes, procs_per_node = mpi_resources.task_partition(
                num_procs, num_nodes, procs_per_node, machinefile
            )

        # Remove portable variable if in extra_args
        if extra_args:
            num_procs, num_nodes, procs_per_node = self._rm_replicated_args(
                num_procs, num_nodes, procs_per_node, p_args
            )

        return {
            "num_procs": num_procs,
            "num_nodes": num_nodes,
            "procs_per_node": procs_per_node,
            "extra_args": extra_args,
            "machinefile": machinefile,
            "hostlist": hostlist,
        }


class MPICH_MPIRunner(MPIRunner):
    def __init__(self, run_command="mpirun"):
        self.run_command = run_command
        self.subgroup_launch = True
        self.mfile_support = True
        self.arg_nprocs = ("-n", "-np")
        self.arg_nnodes = ("--LIBE_NNODES_ARG_EMPTY",)
        self.arg_ppn = ("--ppn",)
        self.mpi_command = [
            self.run_command,
            "--env {env}",
            "-machinefile {machinefile}",
            "-hosts {hostlist}",
            "-np {num_procs}",
            "--ppn {procs_per_node}",
            "{extra_args}",
        ]


class OPENMPI_MPIRunner(MPIRunner):
    def __init__(self, run_command="mpirun"):
        self.run_command = run_command
        self.subgroup_launch = True
        self.mfile_support = True
        self.arg_nprocs = ("-n", "-np", "-c", "--n")
        self.arg_nnodes = ("--LIBE_NNODES_ARG_EMPTY",)
        self.arg_ppn = ("-npernode",)
        self.mpi_command = [
            self.run_command,
            "-x {env}",
            "-machinefile {machinefile}",
            "-host {hostlist}",
            "-np {num_procs}",
            "-npernode {procs_per_node}",
            "{extra_args}",
        ]

    def express_spec(
        self, task, num_procs, num_nodes, procs_per_node, machinefile, hyperthreads, extra_args, resources, workerID
    ):

        hostlist = None
        machinefile = None
        # Use machine files for OpenMPI
        # as "-host" requires entry for every rank

        machinefile = "machinefile_autogen"
        if workerID is not None:
            machinefile += f"_for_worker_{workerID}"
        machinefile += f"_task_{task.id}"
        mfile_created, num_procs, num_nodes, procs_per_node = mpi_resources.create_machinefile(
            resources, machinefile, num_procs, num_nodes, procs_per_node, hyperthreads
        )
        jassert(mfile_created, "Auto-creation of machinefile failed")

        return hostlist, machinefile


class APRUN_MPIRunner(MPIRunner):
    def __init__(self, run_command="aprun"):
        self.run_command = run_command
        self.subgroup_launch = False
        self.mfile_support = False
        self.arg_nprocs = ("-n",)
        self.arg_nnodes = ("--LIBE_NNODES_ARG_EMPTY",)
        self.arg_ppn = ("-N",)
        self.mpi_command = [
            self.run_command,
            "-e {env}",
            "-L {hostlist}",
            "-n {num_procs}",
            "-N {procs_per_node}",
            "{extra_args}",
        ]


class MSMPI_MPIRunner(MPIRunner):
    def __init__(self, run_command="mpiexec"):
        self.run_command = run_command
        self.subgroup_launch = False
        self.mfile_support = False
        self.arg_nprocs = ("-n", "-np")
        self.arg_nnodes = ("--LIBE_NNODES_ARG_EMPTY",)
        self.arg_ppn = ("-cores",)
        self.mpi_command = [
            self.run_command,
            "-env {env}",
            "-n {num_procs}",
            "-cores {procs_per_node}",
            "{extra_args}",
        ]


class SRUN_MPIRunner(MPIRunner):
    def __init__(self, run_command="srun"):
        self.run_command = run_command
        self.subgroup_launch = False
        self.mfile_support = False
        self.arg_nprocs = ("-n", "--ntasks")
        self.arg_nnodes = ("-N", "--nodes")
        self.arg_ppn = ("--ntasks-per-node",)
        self.mpi_command = [
            self.run_command,
            "-w {hostlist}",
            "--ntasks {num_procs}",
            "--nodes {num_nodes}",
            "--ntasks-per-node {procs_per_node}",
            "{extra_args}",
        ]


class JSRUN_MPIRunner(MPIRunner):
    def __init__(self, run_command="jsrun"):
        self.run_command = run_command
        self.subgroup_launch = True
        self.mfile_support = False

        # TODO: Add multiplier to resources checks (for -c/-a)
        self.arg_nprocs = ("--np", "-n")
        self.arg_nnodes = ("--LIBE_NNODES_ARG_EMPTY",)
        self.arg_ppn = ("-r",)
        self.mpi_command = [self.run_command, "-n {num_procs}", "-r {procs_per_node}", "{extra_args}"]

    def get_mpi_specs(
        self, task, num_procs, num_nodes, procs_per_node, machinefile, hyperthreads, extra_args, resources, workerID
    ):

        # Return auto_resource variables inc. extra_args additions
        if extra_args:
            num_procs, num_nodes, procs_per_node, p_args = self._parse_extra_args(
                num_procs, num_nodes, procs_per_node, hyperthreads, extra_args=extra_args
            )

        rm_rpn = True if procs_per_node is None and num_nodes is None else False

        hostlist = None
        if machinefile and not self.mfile_support:
            logger.warning(f"User machinefile ignored - not supported by {self.run_command}")
            machinefile = None
        if machinefile is None and resources is not None:
            num_procs, num_nodes, procs_per_node = mpi_resources.get_resources(
                resources, num_procs, num_nodes, procs_per_node, hyperthreads
            )

            # TODO: Create ERF file if mapping worker to resources req.
        else:
            num_procs, num_nodes, procs_per_node = mpi_resources.task_partition(
                num_procs, num_nodes, procs_per_node, machinefile
            )

        # Remove portable variable if in extra_args
        if extra_args:
            num_procs, num_nodes, procs_per_node = self._rm_replicated_args(
                num_procs, num_nodes, procs_per_node, p_args
            )

        if rm_rpn:
            procs_per_node = None

        return {
            "num_procs": num_procs,
            "num_nodes": num_nodes,
            "procs_per_node": procs_per_node,
            "extra_args": extra_args,
            "machinefile": machinefile,
            "hostlist": hostlist,
        }
