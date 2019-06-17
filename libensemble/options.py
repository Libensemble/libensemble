from libensemble.tests.regression_tests.common import parse_args
from libensemble import libE_logger
import datetime

# Maybe __init__ should automatically call parse_args() ?


class GlobalOptions:
    """The GlobalOptions Class provides methods for managing a global options dictionary.

    Class Attributes:
    -----------------

    current_options_object: object
        Reference to current GlobalOptions object. Modules
        throughout libEnsemble can access the same GlobalOptions object by calling
        the class method current_options()


    Attributes
    ----------
    options: dictionary
        options dictionary storing global options for libEnsemble

    libE_specs: dictionary
        a subset of the global options dictionary currently expected by most
        libEnsemble modules. Largely populated through the regression_tests.common parse_args()
        function
    """

    current_options_object = None

    def __init__(self, *args, **kwargs):
        """ Initializes options to any combination of dictionaries or keywords """
        self.options = {}
        GlobalOptions.current_options_object = self
        for arg in args:
            self.set(arg)
        self.set(kwargs)

    def __str__(self):
        return str(self.options)

    def get(self, *opts):
        """
        Returns any number of matching option entries based on arguments

        Parameters
        ----------

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
            assert isinstance(arg, dict), "Argument isn't a dictionary"
            self.options.update(arg)
        self.options.update(kwargs)

    def parse_args(self):
        """
        Functionality of regression_tests.common parse_args in a more
        natural spot.

        Returns
        -------

        is_master: bool
            Informs current process if it is the master process.

        """
        nworkers, is_master, self.libE_specs, _ = parse_args()
        self.set({'nworkers': nworkers},
                 self.libE_specs)
        return is_master

    def get_libE_specs(self):
        """ Get traditional libE_specs subset """
        # TODO: Investigate methods of parsing out typical libE_specs subsets without
        #   using parse_args()
        return self.libE_specs

    def to_file(self, filename=None):
        """ Save the currently set options to a file.

        Parameters
        ----------
        filename: string
            Requested filename for saved options.
        """

        if not filename:
            filename = self.get('comms') + \
                '_' + datetime.datetime.today().strftime('%d-%m-%Y-%H:%M') \
                + '_options.conf'
        outd = self.options.copy()
        if 'comm' in self.options:
            outd.pop('comm')  # Dealing with comm object reference in from_file() huge pain
        with open(filename, 'w') as f:
            f.write(str(outd))
        f.close()
        return filename

    def from_file(self, filename):
        """ Populates options (except comm) from saved options file.

        Parameters
        ----------
        filename: string
            filename for previously saved options
        """
        import ast
        with open(filename, 'r') as f:
            self.set(ast.literal_eval(f.readline()))
        f.close()


    def current_options():
        """ Class method for other modules to access the most recently defined
        options class instance. """

        return GlobalOptions.current_options_object

    def _check_options():
        """ Check case and consistency of options dictionary """
        pass

    # -----
    # Maybe this isn't the right approach, or the intended approach for logging.
    #   we can already get logger instances from the logger module, whenever we want.

    @staticmethod
    def get_libE_logger():
        """ Return the logger object to the user"""
        return libE_logger

    # @staticmethod
    # def log_get_level():
    #     """ Get libEnsemble logger level """
    #     return libE_logger.get_level()
    #
    # @staticmethod
    # def log_set_filename(filename):
    #     """ Set output filename for libEnsemble's logger"""
    #     libE_logger.set_filename(filename)
    #
    # @staticmethod
    # def log_set_level(level):
    #     """ Set libEnsemble logger level """
    #     libE_logger.set_level(level)
