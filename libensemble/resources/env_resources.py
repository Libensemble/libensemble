"""
This module stores environment variables for use in resource detection
"""

import logging
import os
import re
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)


class EnvResources:
    """Stores environment variables to query for system resource information

    **Class Attributes:**

    :cvar string default_nodelist_env_slurm: Default SLRUM nodelist environment variable
    :cvar string default_nodelist_env_cobalt: Default Cobalt nodelist environment variable
    :cvar string default_nodelist_env_pbs: Default PBS nodelist environment variable (points to nodefile)
    :cvar string default_nodelist_env_lsf: Default LSF nodelist environment variable
    :cvar string default_nodelist_env_lsf_shortform: Default LSF short-form nodelist environment variable

    **Object Attributes:**

    These are set on initialization.

    :ivar dict nodelists: Environment variable names to query for nodelists by scheduler
    :ivar dict ndlist_funcs: Functions to extract nodelists from environment by scheduler
    """

    default_nodelist_env_slurm = "SLURM_NODELIST"
    default_nodelist_env_cobalt = "COBALT_PARTNAME"
    default_nodelist_env_pbs = "PBS_NODEFILE"
    default_nodelist_env_lsf = "LSB_HOSTS"
    default_nodelist_env_lsf_shortform = "LSB_MCPU_HOSTS"

    def __init__(
        self,
        nodelist_env_slurm: str | None = None,
        nodelist_env_cobalt: str | None = None,
        nodelist_env_pbs: str | None = None,
        nodelist_env_lsf: str | None = None,
        nodelist_env_lsf_shortform: str | None = None,
    ) -> None:
        """Initializes a new EnvResources instance

        Determines the environment variables to query for resource
        information. These are either provided or given defaults.

        Parameters
        ----------

        nodelist_env_slurm: String, optional
            The environment variable giving a node list in Slurm format (Default: uses SLURM_NODELIST).
            Note: This is queried only if a node_list file is not provided.

        nodelist_env_cobalt: String, optional
            The environment variable giving a node list in Cobalt format (Default: uses COBALT_PARTNAME).
            Note: This is queried only if a node_list file is not provided and disable_resource_manager=True.

        nodelist_env_pbs: String, optional
            The environment variable giving a path to a nodefile in PBS format (Default: uses PBS_NODEFILE).
            This is queried if a node_list file is not provided.

        nodelist_env_lsf: String, optional
            The environment variable giving a node list in LSF format (Default: uses LSB_HOSTS).
            Note: This is queried only if a node_list file is not provided.

        nodelist_env_lsf_shortform: String, optional
            The environment variable giving a node list in LSF short-form format (Default: uses LSB_MCPU_HOSTS).
            Note: This is queried only if a node_list file is not provided.
        """

        self.scheduler = None
        self.nodelists = {}
        self.nodelists["Slurm"] = nodelist_env_slurm or EnvResources.default_nodelist_env_slurm
        self.nodelists["Cobalt"] = nodelist_env_cobalt or EnvResources.default_nodelist_env_cobalt
        self.nodelists["PBS"] = nodelist_env_pbs or EnvResources.default_nodelist_env_pbs
        self.nodelists["LSF"] = nodelist_env_lsf or EnvResources.default_nodelist_env_lsf
        self.nodelists["LSF_shortform"] = nodelist_env_lsf_shortform or EnvResources.default_nodelist_env_lsf_shortform

        self.ndlist_funcs = {}
        self.ndlist_funcs["Slurm"] = EnvResources.get_slurm_nodelist
        self.ndlist_funcs["Cobalt"] = EnvResources.get_cobalt_nodelist
        self.ndlist_funcs["PBS"] = EnvResources.get_pbs_nodelist
        self.ndlist_funcs["LSF"] = EnvResources.get_lsf_nodelist
        self.ndlist_funcs["LSF_shortform"] = EnvResources.get_lsf_nodelist_frm_shortform

        for env, env_var in self.nodelists.items():
            if os.environ.get(env_var):
                self.scheduler = env
                break

    def get_nodelist(self) -> list[str | Any]:
        """Returns nodelist from environment or an empty list"""
        if self.scheduler:
            env = self.scheduler
            env_var = self.nodelists[env]
            logger.debug(f"{env} env found - getting nodelist from {env_var}")
            get_list_func = self.ndlist_funcs[env]
            global_nodelist = get_list_func(env_var)
            return global_nodelist
        return []

    @staticmethod
    def abbrev_nodenames(node_list: list[str], prefix: str = None) -> list[str]:
        """Returns nodelist with only string up to first dot"""
        newlist = [s.split(".", 1)[0] for s in node_list]
        return newlist

    @staticmethod
    def cobalt_abbrev_nodenames(node_list: list[str], prefix: str = "nid") -> list[str]:
        """Returns nodelist with prefix and leading zeros stripped"""
        newlist = [s.lstrip(prefix) for s in node_list]
        newlist = [s.lstrip("0") for s in newlist]
        return newlist

    def shortnames(self, node_list: list[str]) -> list[str]:
        """Returns nodelist with entries in abbreviated form"""
        if self.scheduler == "Cobalt":
            return EnvResources.cobalt_abbrev_nodenames(node_list)
        elif self.scheduler is not None:
            return EnvResources.abbrev_nodenames(node_list)
        return node_list

    @staticmethod
    def _range_split(s: str) -> tuple[int, int, int]:
        """Splits ID range string"""
        ab = s.split("-", 1)
        nnum_len = len(ab[0])
        a = int(ab[0])
        b = int(ab[-1])
        if a > b:
            a, b = b, a
        b = b + 1
        return a, b, nnum_len

    @staticmethod
    def _noderange_append(prefix: str, nidstr: str, suffix: str) -> list[str]:
        """Formats and appends nodes to overall nodelist"""
        nidlst = []
        for nidgroup in nidstr.split(","):
            a, b, nnum_len = EnvResources._range_split(nidgroup)
            for nid in range(a, b):
                nidlst.append(prefix + str(nid).zfill(nnum_len) + suffix)
        return nidlst

    @staticmethod
    def get_slurm_nodelist(node_list_env: str) -> list[str | Any]:
        """Gets global libEnsemble nodelist from the Slurm environment"""
        fullstr = os.environ[node_list_env]
        if not fullstr:
            return []
        # Split at commas outside of square brackets
        r = re.compile(r"(?:[^,\[]|\[[^\]]*\])+")
        part_splitstr = r.findall(fullstr)
        nidlst = []
        for i in range(len(part_splitstr)):
            part = part_splitstr[i]
            splitstr = part.split("[", 1)
            if len(splitstr) == 1:
                nidlst.append(splitstr[0])
            else:
                prefix = splitstr[0]
                remainder = splitstr[1]
                splitstr = remainder.split("]", 1)
                nidstr = splitstr[0]
                suffix = splitstr[1]
                nidlst.extend(EnvResources._noderange_append(prefix, nidstr, suffix))
        return sorted(nidlst)

    @staticmethod
    def get_cobalt_nodelist(node_list_env: str) -> list[str | Any]:
        """Gets global libEnsemble nodelist from the Cobalt environment"""
        nidlst = []
        nidstr = os.environ[node_list_env]
        if not nidstr:
            return []
        for nidgroup in nidstr.split(","):
            a, b, _ = EnvResources._range_split(nidgroup)
            for nid in range(a, b):
                nidlst.append(str(nid))
        return sorted(nidlst, key=int)

    @staticmethod
    def get_pbs_nodelist(node_list_env: str) -> list[str | Any]:
        """Gets global libEnsemble nodelist path from PBS environment"""
        nidstr_path = os.environ[node_list_env]
        if not nidstr_path:
            return []
        try:
            with open(nidstr_path, "r") as f:
                lines = f.readlines()
        except FileNotFoundError:
            return []
        node_list = [nid.strip("\n") for nid in lines]
        unique_nodelist = list(OrderedDict.fromkeys(node_list))
        unique_nodelist_shortnames = EnvResources.abbrev_nodenames(unique_nodelist)
        return unique_nodelist_shortnames

    @staticmethod
    def get_lsf_nodelist(node_list_env: str) -> list[str | Any]:
        """Gets global libEnsemble nodelist from the LSF environment"""
        full_list = os.environ[node_list_env]
        entries = full_list.split()
        # unique_entries = list(set(entries)) # This will not retain order
        unique_entries = list(OrderedDict.fromkeys(entries))
        nodes = [n for n in unique_entries if "batch" not in n]
        return nodes

    @staticmethod
    def get_lsf_nodelist_frm_shortform(node_list_env: str) -> list[str | Any]:
        """Gets global libEnsemble nodelist from the LSF environment from short-form version"""
        full_list = os.environ[node_list_env]
        entries = full_list.split()
        iter_list = iter(entries)
        zipped_list = list(zip(iter_list, iter_list))
        nodes_with_count = [n for n in zipped_list if "batch" not in n[0]]
        nodes = [n[0] for n in nodes_with_count]
        return nodes
