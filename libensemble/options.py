import os
import logging
from libensemble.comms.logs import LogConfig

class Options:
    """The Options Class provides methods for managing the global options dictionary.
    Also contains method for returning relevant attributes in the form of libE_specs for compatibility.
    Can receive new options as keyword arguments, a single dictionary, or multiple dictionaries.


    Attributes
    ----------
    options: dictionary
        options dictionary storing global options for libEnsemble
    """

    def __init__(self, *args, **kwargs):
        """ Initializes options to any combination of dictionaries or keywords """
        self.options = {}
        for arg in args:
            self.options.update(arg)
        self.options.update(kwargs)

    def __str__(self):
        return str(self.options)

    def set_logging_level(self, level):
    """
    From libensemble.comms.logs. NEEDS TESTING

    Parameters
    ----------
    level: string
        Logging level to set.
    """
        numeric_level = getattr(logging, level.upper(), 10)
        self.log_level = numeric_level
        if self.logger_set:
            logger = logging.getLogger(self.name)
            logger.setLevel(self.log_level)

    def get_options(self, *opts):
        """
        No arguments: all Options dictionary entries.
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

    def set_options(self, new, **kwargs):
        assert type(new) == 'dict'
        self.options.update(new) if new \
        else self.options.update(**kwargs)

    def as_libE_specs():
        pass

    def to_python():
        pass
