# from libensemble import libE_logger
import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--comms', type=str, nargs='?',
                    choices=['local', 'tcp', 'ssh', 'client', 'mpi'],
                    default='local', help='Type of communicator')
parser.add_argument('--nworkers', type=int, nargs='?',
                    help='Number of local forked processes')


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
            return value
        else:
            self.ret_opts = {}
            for arg in opts:
                self.ret_opts.update({arg: self.options.get(arg)})
            return self.ret_opts

    def set(self, *args, **kwargs):
        """ Set any number of new options entries """
        for arg in args:
            assert isinstance(arg, dict), "Argument isn't a dictionary"
            self.options.update(arg)
        self.options.update(kwargs)
        self._check_options()

    def parse_args(self):
        """
        Parse some options from the command-line into options
        """
        args = parser.parse_args()
        self.set(args)

    def get_libE_specs(self):
        """ Get traditional libE_specs subset """
        # TODO: Add support for more libE_specs communication formats
        # There has to be a better way of doing this
        comms = self.get('comms')
        assert comms in ['mpi', 'local'], "Unsupported comms type"
        if comms == 'mpi':
            return self.get('comms', 'comm', 'color')
        elif comms == 'local':
            return self.get('comms', 'nprocesses')

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
        if 'comm' in self.get():
            outd.pop('comm')  # Dealing with comm object reference in from_file() huge pain
        with open(filename, 'w') as f:
            f.write(str(outd))
        f.close()
        return filename

    def from_file(self, filename):
        """ Populates options from saved options file.

        Parameters
        ----------
        filename: string
            filename for previously saved options
        """
        import ast
        with open(filename, 'r') as f:
            self.set(ast.literal_eval(f.readline()))  # Safer way of doing this?
        f.close()

    @staticmethod
    def current_options():
        """ Class method for other modules to access the most recently defined
        options class instance. """

        return GlobalOptions.current_options_object

    def _check_options(self):
        """ Check consistency, MPI settings, logging, and other factors """
        assert isinstance(self.options, dict), "The options aren't a dictionary"
        if len(self.get()):
            self._check_format()
            comms = self.get('comms')
            if comms == 'local':
                self._check_local()
            elif comms == 'mpi':
                self._check_MPI()

    def _check_format(self):
        """ Avoid nested or weirdly formatted settings """
        for i in self.get().items():
            assert len(i) == 2

    def _check_local(self):
        """ Ensure 'local' defaults are set """
        if self.get('nprocesses') is None:
            self.set(nprocesses=4)

    def _check_MPI(self):
        """ Checks MPI options components, populates defaults if necessary"""
        assert self.get('comm') is not None, "MPI Communicator not specified in options"
        if self.get('color') is None:
            self.set(color=0)

    # def _check_logging(self):
    #     pass

    # @staticmethod
    # def get_libE_logger():
    #     """ Return the logger object to the user"""
    #     return libE_logger

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
