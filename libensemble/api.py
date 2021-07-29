import yaml
import importlib
from libensemble.libE import libE
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from libensemble import logger


class Persis_Info:
    def __init__(self, nworkers):
        self.persis_info = {}
        self.nworkers = nworkers

    def add_random_streams(self, explicit_num=None, seed=''):
        if explicit_num:
            num_streams = explicit_num
        else:
            num_streams = self.nworkers + 1

        self.persis_info = add_unique_random_streams({}, num_streams, seed='')


class ExperimentalAPI:
    def __init__(self):
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

    def run(self):
        self.H,
        self.persis_info.persis_info,
        self.flag = libE(self.sim_specs, self.gen_specs,
                         self.exit_criteria,
                         persis_info=self.persis_info.persis_info,
                         alloc_specs=self.alloc_specs,
                         libE_specs=self.libE_specs,
                         H0=self.H0)

    @staticmethod
    def _get_func(specs, type):
        func_path_split = specs[type + '_specs']['function'].rsplit('.', 1)
        return getattr(importlib.import_module(func_path_split[0]), func_path_split[-1])

    @staticmethod
    def _get_inputs(specs, type):
        return [i for i in specs[type + '_specs'].get('inputs', [])]

    @staticmethod
    def _get_outputs(specs, type):
        outputs = specs[type + '_specs'].get('outputs')
        fields = [i for i in outputs]
        field_params = [i for i in outputs.values()]
        results = []
        for i in range(len(fields)):
            built_in_type = __builtins__.get(field_params[i]['type'])
            try:
                if field_params[i]['size'] == 1:
                    size = (1,)
                else:
                    size = field_params[i]['size']
                results.append((fields[i], built_in_type, size))
            except KeyError:
                results.append((fields[i], built_in_type))
        return results

    def from_yaml(self, file):
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
        save_libE_output(self.H, self.persis_info, file, self.nworkers)
