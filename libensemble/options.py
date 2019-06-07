# import os
# import logging
# from libensemble.comms.logs import LogConfig
# from libensemble.tests.regression_tests.common import parse_args
# ^^^ might be useful eventually


class Options:
    """The Options Class provides methods for managing the global options dictionary.
    Also contains method for returning relevant attributes in the form of libE_specs
    for compatibility. Can receive new options as keyword arguments, a single
    dictionary, or multiple dictionaries.


    Attributes
    ----------
    options: dictionary
        options dictionary storing global options for libEnsemble
    """

    def __init__(self, *args, **kwargs):
        """ Initializes options to any combination of dictionaries or keywords """
        # self.nworkers, self.is_master, self.libE_specs, _ = parse_args()
        self.options = {}
        for arg in args:        # Duplicate of code in set_options()
            self.options.update(arg)
        self.options.update(kwargs)

    def __str__(self):
        return str(self.options)

    def get_options(self, *opts):
        """
        No arguments: all dictionary entries.
        1 argument: corresponding dictionary value matching argument key
        2+ arguments: dictionary with entries matching arguments
        """
        if not opts:
            return self.options
        elif len(opts) == 1:
            value = self.options.get(opts[0])
            assert 'None' not in value, 'Requested option not found'
            return value
        else:
            self.ret_opts = {}
            for arg in opts:
                self.ret_opts.update({arg: self.options.get(arg)})
            assert None not in self.ret_opts.values(), 'Requested option not found'
            return self.ret_opts

    def set_options(self, *args, **kwargs):
        for arg in args:
            self.options.update(arg)
        self.options.update(kwargs)

    def as_libE_specs():
        pass

    def to_python_config():
        pass
