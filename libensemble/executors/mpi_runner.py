import argparse
import logging

from libensemble.executors.executor import jassert
from libensemble.resources import mpi_resources

logger = logging.getLogger(__name__)
# To change logging level for just this module
# logger.setLevel(logging.DEBUG)


class MPIRunner:
    @staticmethod
    def get_runner(mpi_runner_type, runner_name=None, platform_info=None):
        mpi_runners = {
            "mpich": MPICH_MPIRunner,
            "openmpi": OPENMPI_MPIRunner,
            "aprun": APRUN_MPIRunner,
            "srun": SRUN_MPIRunner,
            "jsrun": JSRUN_MPIRunner,
            "msmpi": MSMPI_MPIRunner,
            "custom": MPIRunner,
        }
        runner = None
        if mpi_runner_type is not None:
            mpi_runner = mpi_runners[mpi_runner_type]
            if runner_name is not None:
                runner = mpi_runner(run_command=runner_name, platform_info=platform_info)
            else:
                runner = mpi_runner(platform_info=platform_info)
        return runner

    def __init__(self, run_command="mpiexec", platform_info=None):
        self.run_command = run_command
        self.mpi_command = [self.run_command, "{extra_args}"]
        self.subgroup_launch = False
        self.mfile_support = False
        self.arg_nprocs = ("--LIBE_NPROCS_ARG_EMPTY",)
        self.arg_nnodes = ("--LIBE_NNODES_ARG_EMPTY",)
        self.arg_ppn = ("--LIBE_PPN_ARG_EMPTY",)
        self.default_mpi_options = None
        self.default_gpu_args = None
        self.default_gpu_arg_type = None
        self.platform_info = platform_info
        self.rm_rpn = False

    def _get_parser(self, p_args, nprocs, nnodes, ppn):
        """Parses MPI arguments from the provided string"""
        parser = argparse.ArgumentParser(description="Parse extra_args", allow_abbrev=False)
        parser.add_argument(*nprocs, type=int, dest="num_procs", default=None)
        parser.add_argument(*nnodes, type=int, dest="num_nodes", default=None)
        parser.add_argument(*ppn, type=int, dest="procs_per_node", default=None)
        args, _ = parser.parse_known_args(p_args)
        return args

    def _parse_extra_args(self, nprocs, nnodes, ppn, hyperthreads, extra_args):
        """Fill in missing portable MPI options from extra_args string

        These can be used in resource checking in ``mpi_resources.get_resources``
        """
        splt_extra_args = extra_args.split()
        p_args = self._get_parser(splt_extra_args, self.arg_nprocs, self.arg_nnodes, self.arg_ppn)

        # Only fill from extra_args if not set by portable options
        if nprocs is None:
            nprocs = p_args.num_procs
        if nnodes is None:
            nnodes = p_args.num_nodes
        if ppn is None:
            ppn = p_args.procs_per_node

        extra_args = " ".join(splt_extra_args)
        return nprocs, nnodes, ppn, p_args

    def _rm_replicated_args(self, nprocs, nnodes, ppn, p_args):
        """Removed replicated arguments.

        To be called after ``mpi_resources.get_resources``
        """
        if p_args is not None:
            if p_args.num_procs is not None:
                nprocs = None
            if p_args.num_nodes is not None:
                nnodes = None
            if p_args.procs_per_node is not None:
                ppn = None
        return nprocs, nnodes, ppn

    def _append_to_extra_args(self, extra_args, new_args):
        """Add a string to extra_args"""
        if extra_args is None:
            return new_args
        extra_args += f" {new_args}"
        return extra_args

    def express_spec(self, task, nprocs, nnodes, ppn, machinefile, hyperthreads, extra_args, resources, workerID):
        """Returns a hostlist or machinefile name

        If a machinefile is used, the file will also be created. This function
        is designed to be overridden by inheritance.
        """
        hostlist = None
        machinefile = None
        # Always use host lists (unless uneven mapping)
        hostlist = mpi_resources.get_hostlist(resources, nnodes)
        return hostlist, machinefile

    def _set_gpu_cli_option(self, wresources, extra_args, gpu_setting_name, gpu_value):
        """Update extra args with the GPU setting for the MPI runner"""
        jassert(wresources.even_slots, f"Cannot assign CPUs/GPUs to uneven slots per node {wresources.slots}")

        if gpu_setting_name.endswith("="):
            gpus_opt = gpu_setting_name + str(gpu_value)
        else:
            gpus_opt = gpu_setting_name + " " + str(gpu_value)

        if extra_args is None:
            extra_args = gpus_opt
        else:
            extra_args = " ".join((extra_args, gpus_opt))
        return extra_args

    def _set_gpu_env_var(self, wresources, task, gpus_per_node, gpus_env):
        """Add GPU environment variable setting to the tasks environment"""
        jassert(wresources.matching_slots, f"Cannot assign CPUs/GPUs to non-matching slots per node {wresources.slots}")
        slot_list = wresources.get_slots_as_string(multiplier=wresources.gpus_per_rset_per_node, limit=gpus_per_node)
        task._add_to_env(gpus_env, slot_list)

    def _local_runner_set_gpus(self, task, wresources, extra_args, gpus_per_node, ppn):
        """Set default GPU setting for MPI runner"""

        arg_type = self.default_gpu_arg_type
        if arg_type is not None:
            gpu_value = gpus_per_node // ppn if arg_type == "option_gpus_per_task" else gpus_per_node
            gpu_setting_name = self.default_gpu_args[arg_type]
            jassert(gpu_setting_name is not None, f"No default gpu_setting_name for {arg_type}")
            extra_args = self._set_gpu_cli_option(wresources, extra_args, gpu_setting_name, gpu_value)
        else:
            gpus_env = "CUDA_VISIBLE_DEVICES"
            self._set_gpu_env_var(wresources, task, gpus_per_node, gpus_env)
        return extra_args

    def _get_default_arg(self, gpu_setting_type):
        """Return default setting for the given gpu_setting_type if it exists, else error"""
        jassert(
            gpu_setting_type in ["option_gpus_per_node", "option_gpus_per_task"],
            f"Unrecognized gpu_setting_type {gpu_setting_type}",
        )
        jassert(
            self.default_gpu_args is not None,
            "The current MPI runner has no default command line option for setting GPUs",
        )
        gpu_setting_name = self.default_gpu_args[gpu_setting_type]
        jassert(gpu_setting_name is not None, f"No default GPU setting for {gpu_setting_type}")
        return gpu_setting_name

    def _assign_gpus(self, task, resources, nprocs, nnodes, ppn, ngpus, extra_args, match_procs_to_gpus):
        """Assign GPU resources to slots, limited by ngpus if present.

        GPUs will be assigned using the slot count and GPUs per slot (from resources).
        If ``match_procs_to_gpus`` is True, then MPI processor/node configuration will
        be added to match the GPU setting.

        The method used to assign GPUs will be determined either by platform settings
        or the default for the MPI runner.

        Returns updated MPI configuration variables, and updates the task environment
        attribute.

        """

        wresources = resources.worker_resources

        # gpus per node for this worker.
        if wresources.doihave_gpus():
            gpus_avail_per_node = wresources.slot_count * wresources.gpus_per_rset_per_node
        else:
            gpus_avail_per_node = 0

        if nnodes is None:
            if nprocs:
                if ppn:
                    nnodes = nprocs // ppn
                else:
                    nnodes = min(nprocs, wresources.local_node_count)
            else:
                nnodes = wresources.local_node_count

        if ngpus is not None:
            gpus_req_per_node = ngpus // nnodes
            if gpus_req_per_node > gpus_avail_per_node:
                logger.info(f"Asked for more GPUs per node than available - max is {gpus_avail_per_node}")
            gpus_per_node = min(gpus_req_per_node, gpus_avail_per_node)
        else:
            gpus_per_node = gpus_avail_per_node
        task.ngpus_req = gpus_per_node

        if match_procs_to_gpus:
            ppn = gpus_per_node
            nprocs = nnodes * ppn
            jassert(nprocs > 0, f"Matching procs to GPUs has resulted in {nprocs} procs")

        if ngpus == 0:
            # if request zero gpus, return here
            return nprocs, nnodes, ppn, extra_args

        gpu_setting_type = "runner_default"

        if ppn is None:
            ppn = nprocs // nnodes

        if self.platform_info:
            gpu_setting_type = self.platform_info.get("gpu_setting_type", gpu_setting_type)

        if gpu_setting_type == "runner_default":
            extra_args = self._local_runner_set_gpus(task, wresources, extra_args, gpus_per_node, ppn)

        elif gpu_setting_type in ["option_gpus_per_node", "option_gpus_per_task"]:
            gpu_value = gpus_per_node // ppn if gpu_setting_type == "option_gpus_per_task" else gpus_per_node
            gpu_setting_name = self.platform_info.get("gpu_setting_name", self._get_default_arg(gpu_setting_type))
            extra_args = self._set_gpu_cli_option(wresources, extra_args, gpu_setting_name, gpu_value)

        else:  # gpu_setting_type == "env":
            gpus_env = self.platform_info.get("gpu_setting_name", "CUDA_VISIBLE_DEVICES")
            self._set_gpu_env_var(wresources, task, gpus_per_node, gpus_env)

        return nprocs, nnodes, ppn, extra_args

    def _get_min_nodes(self, nprocs, ppn, nnodes, ngpus, resources):
        """Get minimum nodes needed to match configuration"""
        if nnodes is not None:
            return nnodes
        if ppn:
            return None  # nnodes gets processed later.
        if resources is not None:
            wresources = resources.worker_resources
            total_nodes = wresources.local_node_count
            procs_on_node = wresources.slot_count * wresources.procs_per_rset_per_node

            if not nprocs and ngpus is None:
                # Delay node evaluation to GPU assignment code
                return None
            proc_min_nodes = 1
            gpu_min_nodes = 1
            if nprocs:
                proc_min_nodes = (nprocs + procs_on_node - 1) // procs_on_node
            if ngpus:
                gpus_on_node = wresources.slot_count * wresources.gpus_per_rset_per_node
                gpu_min_nodes = (ngpus + gpus_on_node - 1) // gpus_on_node

            min_nodes = max(proc_min_nodes, gpu_min_nodes)
            nnodes = min(min_nodes, total_nodes)
            # Must have at least one processor per node to use GPUs
            if nprocs:
                nnodes = min(nnodes, nprocs)
            return nnodes

    def _adjust_procs(self, nprocs, ppn, nnodes, ngpus, resources):
        """Adjust an invalid config"""

        def adjust_resource(n_units, units_attr, units_name):
            if n_units is not None and nnodes:
                mod_n_units = n_units % nnodes
                if mod_n_units != 0:
                    try_n_units = n_units + (nnodes - mod_n_units)
                    if try_n_units <= wresources.slot_count * getattr(wresources, units_attr) * nnodes:
                        logger.info(
                            f"Adjusted {units_name} to split evenly across nodes. From {n_units} to {try_n_units}"
                        )
                        return try_n_units
            return n_units

        if resources is not None:
            wresources = resources.worker_resources
            ngpus = adjust_resource(ngpus, "gpus_per_rset_per_node", "ngpus")
            nprocs = adjust_resource(nprocs, "procs_per_rset_per_node", "nprocs")
        return nprocs, ngpus

    def get_mpi_specs(
        self,
        task,
        nprocs,
        nnodes,
        ppn,
        ngpus,
        machinefile,
        hyperthreads,
        extra_args,
        auto_assign_gpus,
        match_procs_to_gpus,
        resources,
        workerID,
    ):
        """Returns a dictionary with the MPI specifications for the runline.

        This function takes user provided inputs and resource information and
        uses these to determine the final MPI specifications. This may include
        a host-list or machine file.

        extra_args will be parsed if possible to extract MPI configuration.
        Default arguments may be added, and GPU settings added to extra_args,
        or to the task environment.
        """

        p_args = None

        # Return auto_resource variables inc. extra_args additions
        if extra_args:
            nprocs, nnodes, ppn, p_args = self._parse_extra_args(
                nprocs, nnodes, ppn, hyperthreads, extra_args=extra_args
            )

        # If no_config_set and auto_assign_gpus - make match_procs_to_gpus default.
        no_config_set = not (nprocs or ppn)

        if match_procs_to_gpus:
            jassert(no_config_set, "match_procs_to_gpus is mutually exclusive with either of nprocs/ppn")

        nnodes = self._get_min_nodes(nprocs, ppn, nnodes, ngpus, resources)
        nprocs, ngpus = self._adjust_procs(nprocs, ppn, nnodes, ngpus, resources)

        if auto_assign_gpus or ngpus is not None:
            # if no_config_set, make match_procs_to_gpus default.
            if no_config_set:
                match_procs_to_gpus = True
            nprocs, nnodes, ppn, extra_args = self._assign_gpus(
                task, resources, nprocs, nnodes, ppn, ngpus, extra_args, match_procs_to_gpus
            )

        rm_rpn = self.rm_rpn and ppn is None and nnodes is None

        hostlist = None
        if machinefile and not self.mfile_support:
            logger.warning(f"User machinefile ignored - not supported by {self.run_command}")
            machinefile = None

        if machinefile is None and resources is not None:
            nprocs, nnodes, ppn = mpi_resources.get_resources(resources, nprocs, nnodes, ppn, hyperthreads)
            hostlist, machinefile = self.express_spec(
                task, nprocs, nnodes, ppn, machinefile, hyperthreads, extra_args, resources, workerID
            )
        else:
            nprocs, nnodes, ppn = mpi_resources.task_partition(nprocs, nnodes, ppn, machinefile)

        # Remove portable variable if in extra_args
        if extra_args:
            nprocs, nnodes, ppn = self._rm_replicated_args(nprocs, nnodes, ppn, p_args)

        if rm_rpn:
            ppn = None

        if self.default_mpi_options is not None:
            extra_args = self._append_to_extra_args(extra_args, self.default_mpi_options)

        return {
            "num_procs": nprocs,
            "num_nodes": nnodes,
            "procs_per_node": ppn,
            "extra_args": extra_args,
            "machinefile": machinefile,
            "hostlist": hostlist,
        }


class MPICH_MPIRunner(MPIRunner):
    def __init__(self, run_command="mpirun", platform_info=None):
        self.run_command = run_command
        self.subgroup_launch = True
        self.mfile_support = True
        self.arg_nprocs = ("-n", "-np")
        self.arg_nnodes = ("--LIBE_NNODES_ARG_EMPTY",)
        self.arg_ppn = ("--ppn", "-ppn")
        self.default_mpi_options = None
        self.default_gpu_args = None
        self.default_gpu_arg_type = None
        self.platform_info = platform_info
        self.rm_rpn = False

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
    def __init__(self, run_command="mpirun", platform_info=None):
        self.run_command = run_command
        self.subgroup_launch = True
        self.mfile_support = True
        self.arg_nprocs = ("-n", "-np", "-c", "--n")
        self.arg_nnodes = ("--LIBE_NNODES_ARG_EMPTY",)
        self.arg_ppn = ("-npernode",)
        self.default_mpi_options = None
        self.default_gpu_args = None
        self.default_gpu_arg_type = None
        self.platform_info = platform_info
        self.rm_rpn = False
        self.mpi_command = [
            self.run_command,
            "-x {env}",
            "-machinefile {machinefile}",
            "-host {hostlist}",
            "-np {num_procs}",
            "-npernode {procs_per_node}",
            "{extra_args}",
        ]

    def express_spec(self, task, nprocs, nnodes, ppn, machinefile, hyperthreads, extra_args, resources, workerID):
        """Returns a hostlist or machinefile name

        If a machinefile is used, the file will also be created. This function
        is designed to be overridden by inheritance.
        """
        hostlist = None
        machinefile = None
        # Use machine files for Open-MPI
        # as "-host" requires entry for every rank

        machinefile = "machinefile_autogen"
        machinefile += f"_for_worker_{workerID}"
        machinefile += f"_task_{task.id}"
        mfile_created, nprocs, nnodes, ppn = mpi_resources.create_machinefile(
            resources, machinefile, nprocs, nnodes, ppn, hyperthreads
        )
        jassert(mfile_created, "Auto-creation of machinefile failed")

        return hostlist, machinefile


class APRUN_MPIRunner(MPIRunner):
    def __init__(self, run_command="aprun", platform_info=None):
        self.run_command = run_command
        self.subgroup_launch = False
        self.mfile_support = False
        self.arg_nprocs = ("-n",)
        self.arg_nnodes = ("--LIBE_NNODES_ARG_EMPTY",)
        self.arg_ppn = ("-N",)
        self.default_mpi_options = None
        self.default_gpu_args = None
        self.default_gpu_arg_type = None
        self.platform_info = platform_info
        self.rm_rpn = False
        self.mpi_command = [
            self.run_command,
            "-e {env}",
            "-L {hostlist}",
            "-n {num_procs}",
            "-N {procs_per_node}",
            "{extra_args}",
        ]


class MSMPI_MPIRunner(MPIRunner):
    def __init__(self, run_command="mpiexec", platform_info=None):
        self.run_command = run_command
        self.subgroup_launch = False
        self.mfile_support = False
        self.arg_nprocs = ("-n", "-np")
        self.arg_nnodes = ("--LIBE_NNODES_ARG_EMPTY",)
        self.arg_ppn = ("-cores",)
        self.default_mpi_options = None
        self.default_gpu_args = None
        self.default_gpu_arg_type = None
        self.platform_info = platform_info
        self.rm_rpn = False
        self.mpi_command = [
            self.run_command,
            "-env {env}",
            "-n {num_procs}",
            "-cores {procs_per_node}",
            "{extra_args}",
        ]


class SRUN_MPIRunner(MPIRunner):
    def __init__(self, run_command="srun", platform_info=None):
        self.run_command = run_command
        self.subgroup_launch = False
        self.mfile_support = False
        self.arg_nprocs = ("-n", "--ntasks")
        self.arg_nnodes = ("-N", "--nodes")
        self.arg_ppn = ("--ntasks-per-node",)
        self.default_mpi_options = "--exact"
        self.default_gpu_arg_type = "option_gpus_per_task"
        self.default_gpu_args = {"option_gpus_per_task": "--gpus-per-task", "option_gpus_per_node": "--gpus-per-node"}
        self.platform_info = platform_info
        self.rm_rpn = False
        self.mpi_command = [
            self.run_command,
            "-w {hostlist}",
            "--ntasks {num_procs}",
            "--nodes {num_nodes}",
            "--ntasks-per-node {procs_per_node}",
            "{extra_args}",
        ]


class JSRUN_MPIRunner(MPIRunner):
    def __init__(self, run_command="jsrun", platform_info=None):
        self.run_command = run_command
        self.subgroup_launch = True
        self.mfile_support = False
        self.arg_nprocs = ("--np", "-n")
        self.arg_nnodes = ("--LIBE_NNODES_ARG_EMPTY",)
        self.arg_ppn = ("-r",)
        self.default_mpi_options = None
        self.default_gpu_arg_type = "option_gpus_per_task"
        self.default_gpu_args = {"option_gpus_per_task": "-g", "option_gpus_per_node": None}

        self.platform_info = platform_info
        self.mpi_command = [self.run_command, "-n {num_procs}", "-r {procs_per_node}", "{extra_args}"]
        self.rm_rpn = True

    def express_spec(self, task, nprocs, nnodes, ppn, machinefile, hyperthreads, extra_args, resources, workerID):
        """Returns None, None as jsrun uses neither hostlist or machinefile"""
        return None, None
