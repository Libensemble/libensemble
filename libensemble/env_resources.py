"""
Module for storing environment variables for use in resource detection

"""

import os
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)


class EnvResources:
    """Store environment variables to query for system resource information"""

    default_nodelist_env_slurm = 'SLURM_NODELIST'
    default_nodelist_env_cobalt = 'COBALT_PARTNAME'
    default_nodelist_env_lsf = 'LSB_HOSTS'
    default_nodelist_env_lsf_shortform = 'LSB_MCPU_HOSTS'

    def __init__(self,
                 nodelist_env_slurm=None,
                 nodelist_env_cobalt=None,
                 nodelist_env_lsf=None,
                 nodelist_env_lsf_shortform=None):

        self.nodelists = {}
        self.nodelists['Slurm'] = nodelist_env_slurm or EnvResources.default_nodelist_env_slurm
        self.nodelists['Cobalt'] = nodelist_env_cobalt or EnvResources.default_nodelist_env_cobalt
        self.nodelists['LSF'] = nodelist_env_lsf or EnvResources.default_nodelist_env_lsf
        self.nodelists['LSF_shortform'] = nodelist_env_lsf_shortform or EnvResources.default_nodelist_env_lsf_shortform

        self.ndlist_funcs = {}
        self.ndlist_funcs['Slurm'] = EnvResources.get_slurm_nodelist
        self.ndlist_funcs['Cobalt'] = EnvResources.get_cobalt_nodelist
        self.ndlist_funcs['LSF'] = EnvResources.get_lsf_nodelist
        self.ndlist_funcs['LSF_shortform'] = EnvResources.get_lsf_nodelist_frm_shortform

    def get_nodelist(self):
        for env, env_var in self.nodelists.items():
            if os.environ.get(env_var):
                logger.debug("{0} env found - getting nodelist from {0}".format(env))
                get_list_func = self.ndlist_funcs[env]
                global_nodelist = get_list_func(env_var)
                return global_nodelist
        return []

    @staticmethod
    def _range_split(s):
        """Split ID range string."""
        ab = s.split("-", 1)
        nnum_len = len(ab[0])
        a = int(ab[0])
        b = int(ab[-1])
        if a > b:
            a, b = b, a
        b = b + 1
        return a, b, nnum_len

    @staticmethod
    def get_slurm_nodelist(node_list_env):
        """Get global libEnsemble nodelist from the Slurm environment"""
        nidlst = []
        fullstr = os.environ[node_list_env]
        if not fullstr:
            return []
        splitstr = fullstr.split('-', 1)
        prefix = splitstr[0]
        nidstr = splitstr[1].strip("[]")
        for nidgroup in nidstr.split(','):
            a, b, nnum_len = EnvResources._range_split(nidgroup)
            for nid in range(a, b):
                nidlst.append(prefix + '-' + str(nid).zfill(nnum_len))
        return sorted(nidlst)

    @staticmethod
    def get_cobalt_nodelist(node_list_env):
        """Get global libEnsemble nodelist from the Cobalt environment"""
        nidlst = []
        nidstr = os.environ[node_list_env]
        if not nidstr:
            return []
        for nidgroup in nidstr.split(','):
            a, b, _ = EnvResources._range_split(nidgroup)
            for nid in range(a, b):
                nidlst.append(str(nid))
        return sorted(nidlst, key=int)

    @staticmethod
    def get_lsf_nodelist(node_list_env):
        """Get global libEnsemble nodelist from the LSF environment"""
        full_list = os.environ[node_list_env]
        entries = full_list.split()
        # unique_entries = list(set(entries)) # This will not retain order
        unique_entries = list(OrderedDict.fromkeys(entries))
        nodes = [n for n in unique_entries if 'batch' not in n]
        return nodes

    @staticmethod
    def get_lsf_nodelist_frm_shortform(node_list_env):
        """Get global libEnsemble nodelist from the LSF environment from short-form version"""
        full_list = os.environ[node_list_env]
        entries = full_list.split()
        iter_list = iter(entries)
        zipped_list = list(zip(iter_list, iter_list))
        nodes_with_count = [n for n in zipped_list if 'batch' not in n[0]]
        nodes = [n[0] for n in nodes_with_count]
        return nodes
