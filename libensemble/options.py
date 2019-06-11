from libensemble.tests.regression_tests.common import parse_args


class Options:
    """The Options Class provides methods for managing a global options dictionary.


    Attributes
    ----------
    options: dictionary
        options dictionary storing global options for libEnsemble

    nworkers: integer
        number of workers initialized to perform work

    libE_specs: dictionary
        a subset of the global options dictionary currently expected by most
        libEnsemble modules. Largely populated through the regression_tests.common parse_args()
        function
    """

    def __init__(self, *args, **kwargs):
        """ Initializes options to any combination of dictionaries or keywords
            Eventually should also initialize to preexisting settings stored somewhere """
        self.options = {}
        for arg in args:
            self.options.update(arg)
        self.options.update(kwargs)

    def __str__(self):
        return str(self.options)

    def parse_args(self):
        """ Include functionality of regression_tests.common parse_args in a more
        natural spot. Adds this information to global options


        """
        self.nworkers, is_master, self.libE_specs, _ = parse_args()
        self.set({'nworkers': self.nworkers},
                 self.libE_specs)
        return is_master

    def get(self, *opts):
        """
        Returns any number of matching option entries based on arguments

        No arguments: all dictionary entries.
        1 argument: corresponding dictionary value matching argument
        2+ arguments: dictionary with entries matching arguments
        """
        if not opts:
            return self.options
        elif len(opts) == 1:
            value = self.options.get(opts[0])
            assert value is not None, 'Requested option not found'
            return value
        else:
            self.ret_opts = {}
            for arg in opts:
                self.ret_opts.update({arg: self.options.get(arg)})
            assert None not in self.ret_opts.values(), 'Requested option not found'
            return self.ret_opts

    def set(self, *args, **kwargs):
        """ Set any number of new options entries """
        for arg in args:
            self.options.update(arg)
        self.options.update(kwargs)

    def get_libe_specs(self):
        """ Get traditional libE_specs subset """
        return self.libE_specs
