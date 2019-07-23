import datetime
import logging

logger = logging.getLogger(__name__)


class GlobalOptions:
    """The GlobalOptions Class provides methods for managing a global options dictionary.

    **Class Attributes:**

    current_options_object: :obj: `GlobalOptions`
        Currently initiated GlobalOptions object

    **Object Attributes:**

    options: :obj: `dict`
        dictionary storing global options

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
        """ Returns any number of matching option entries based on arguments """
        if not opts:
            return self.options
        elif len(opts) == 1:
            value = self.options.get(opts[0])
            return value
        else:
            ret_opts = {}
            for arg in opts:
                ret_opts.update({arg: self.options.get(arg)})
            return ret_opts

    def set(self, *args, **kwargs):
        """ Set any number of new options entries """
        for arg in args:
            assert isinstance(arg, dict), "Argument isn't a dictionary or keyword"
            self.options.update(arg)
        self.options.update(kwargs)
        self._check_options()

    def get_libE_specs(self):
        """ Get traditional libE_specs subset """
        # TODO: Add support for more libE_specs communication formats
        # OR TODO: Restructure libEnsemble to not depend on libE_specs (Options instead)
        #  in which case, this function will no longer be needed
        comms = self.get('comms')
        if comms == 'mpi':
            return self.get('comms', 'comm', 'color')
        elif comms == 'local':
            return self.get('comms', 'nprocesses')
        else:
            logger.warning("Unsupported comms type for libE_specs subset. Returning all options.")
            return self.get()

    def to_file(self, filename=None):
        """ Save the currently set options (except COMM) to a file.

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
            outd.pop('comm')
        with open(filename, 'w') as f:
            for opt in outd:
                outstr = '{} = {}\n'.format(opt.upper(), outd[opt])
                f.write(outstr)
        f.close()
        return filename

    def from_file(self, filename):
        """ Populates options from saved options file.

        Parameters
        ----------
        filename: string
            filename for previously saved options
        """
        indict = {}
        with open(filename, 'r') as f:
            for line in f.readlines():
                entry_pair = line[:-1].split(' = ')
                [key, value] = [ent.lower() for ent in entry_pair]
                indict[key] = int(value) if value.isnumeric() else value
        self.set(indict)
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
        if self.get('comm') is None:
            logger.warning("MPI COMM not detected. Were options just loaded?")
        if self.get('color') is None:
            self.set(color=0)
