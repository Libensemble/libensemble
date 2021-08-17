import yaml
import pprint
import inspect
import importlib
from libensemble.libE import libE
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from libensemble.version import __version__
from libensemble import logger


class Persis_Info:
    """
    ``persis_info`` persistent information dictionary management class.
    An instance of this is created on initiation of ExperimentalAPI, since
    ``persis_info`` is populated for most libEnsemble test-cases anyway.
    """
    def __init__(self, nworkers):
        self.persis_info = {}
        self.nworkers = nworkers

    def add_random_streams(self, explicit_num=None, seed=''):
        """ ``Persis_Info`` wrapper for ``add_unique_random_streams``"""
        if explicit_num:
            num_streams = explicit_num
        else:
            num_streams = self.nworkers + 1

        self.persis_info = add_unique_random_streams({}, num_streams, seed='')


class ExperimentalAPI:
    """
    The vast majority of libEnsemble cases require the user to instantiate
    and populate a set of specification dictionaries, call ``parse_args()``,
    then call ``libE()`` while passing in each spec dictionary. Most ``libE()``
    calls are therefore identical, even across widely varying use-cases. This
    is an alternative interface for parameterizing libEnsemble by interacting
    with a class instance.
    """
    def __init__(self):
        """ Initializes an API instance. ``parse_args() is called on instantiation """
        self.nworkers, self.is_manager, self.libE_specs, _ = parse_args()
        self.persis_info = Persis_Info(self.nworkers)
        self.logger = logger
        self.logger.set_level('INFO')
        self.sim_specs = {}
        self.gen_specs = {}
        self.alloc_specs = {}
        self.exit_criteria = {}
        self.H = None
        self.H0 = None
        self._filename = inspect.stack()[1].filename

    def __str__(self):
        """ Returns a pretty-printed representation of API object """
        info = '\nlibEnsemble {}\n'.format(__version__) + 79*'*' + '\n'
        info += '\nCalling Script: ' + self._filename.split('/')[-1] + '\n'

        dicts = {'libE_specs': self.libE_specs,
                 'sim_specs': self.sim_specs,
                 'gen_specs': self.gen_specs,
                 'alloc_specs': self.alloc_specs,
                 'persis_info': self.persis_info.persis_info,
                 'exit_criteria': self.exit_criteria}

        for i in dicts:
            info += '{}:\n {} \n\n'.format(i, pprint.pformat(dicts[i]))

        info += 79*'*'
        return info


    def run(self):
        """
        Initializes libEnsemble, passes in all specification dictionaries.
        Sets API instance's output H, final persis_info state, and flag.
        """
        self.H, self.persis_info.persis_info, self.flag = \
            libE(self.sim_specs, self.gen_specs, self.exit_criteria,
                 persis_info=self.persis_info.persis_info,
                 alloc_specs=self.alloc_specs,
                 libE_specs=self.libE_specs,
                 H0=self.H0)

    @staticmethod
    def _get_func(specs, type):
        """ Extracts user function specified in specs dict """
        func_path_split = specs[type + '_specs']['function'].rsplit('.', 1)
        return getattr(importlib.import_module(func_path_split[0]), func_path_split[-1])

    @staticmethod
    def _get_inputs(specs, type):
        """ Extracts input parameters from specs dict """
        return [i for i in specs[type + '_specs'].get('inputs', [])]

    @staticmethod
    def _get_outputs(specs, type):
        """ Extracts output parameters from specs dict """
        outputs = specs[type + '_specs'].get('outputs')
        fields = [i for i in outputs]
        field_params = [i for i in outputs.values()]
        results = []
        for i in range(len(fields)):
            built_in_type = __builtins__.get(field_params[i]['type'])
            try:
                if field_params[i]['size'] == 1:
                    size = (1,)  # typically how size 1 is preferred?
                else:
                    size = field_params[i]['size']
                results.append((fields[i], built_in_type, size))
            except KeyError:
                results.append((fields[i], built_in_type))
        return results

    def from_yaml(self, file):
        """ Populates libEnsemble specs dictionaries from yaml file """
        with open(file, 'r') as f:
            specs = yaml.full_load(f)

        self.sim_specs['sim_f'] = self._get_func(specs, 'sim')
        self.gen_specs['gen_f'] = self._get_func(specs, 'gen')
        self.alloc_specs['alloc_f'] = self._get_func(specs, 'alloc')

        self.sim_specs['in'] = self._get_inputs(specs, 'sim')
        self.gen_specs['in'] = self._get_inputs(specs, 'gen')
        self.alloc_specs['in'] = self._get_inputs(specs, 'alloc')

        self.sim_specs['out'] = self._get_outputs(specs, 'sim')
        self.gen_specs['out'] = self._get_outputs(specs, 'gen')
        self.alloc_specs['out'] = self._get_outputs(specs, 'alloc')

        self.sim_specs['user'] = specs['sim_specs'].get('user', {})
        self.gen_specs['user'] = specs['gen_specs'].get('user', {})
        self.alloc_specs['user'] = specs['alloc_specs'].get('user', {})

        self.exit_criteria = specs['libE_specs']['exit_criteria']
        specs['libE_specs'].pop('exit_criteria')

        self.libE_specs.update(specs['libE_specs'])

    def save_output(self, file):
        """ Class wrapper for save_libE_output """
        save_libE_output(self.H, self.persis_info, file, self.nworkers)
