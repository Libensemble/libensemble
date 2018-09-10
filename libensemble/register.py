#!/usr/bin/env python

"""Module to register applications to libEnsemble"""

import os
import logging
from mpi4py import MPI

logger = logging.getLogger(__name__)
#For debug messages in this module  - uncomment
#(see libE.py to change root logging level)
#logger.setLevel(logging.DEBUG)

class RegistrationException(Exception):
    "Raised for any exception in the Register"
    pass

def rassert(check, *args):
    "Version of assert that raises a RegistrationException"
    if not check:
        raise RegistrationException(*args)


class Application:
    """An application is an executable user-program
    (e.g. Implementing a sim/gen function)."""

    def __init__(self, full_path, calc_type='sim', desc=None):
        """Instantiate a new Application instance."""
        self.full_path = full_path
        self.calc_type = calc_type
        self.calc_dir, self.exe = os.path.split(full_path)

        # Use this name to delete jobs in database - see del_apps(), del_jobs()
        self.name = self.exe + '.' + self.calc_type + 'func'
        self.desc = desc or (self.exe + ' ' + self.calc_type + ' function')

#May merge this into job_controller
class Register():
    """Registers and stores user applications

    Attributes
    ----------
    default_registry : Obj: Register or inherited class.
        A class attribute holding the default registry.

    """

    default_registry = None

    @property
    def sim_default_app(self):
        """Return the default simulation app."""
        return self._default_apps['sim']

    @property
    def gen_default_app(self):
        """Return the default generator app."""
        return self._default_apps['gen']

    def default_app(self, calc_type):
        """Return the default calc_type app."""
        return self._default_apps.get(calc_type)

    def __init__(self):
        """Instantiate a new Register instance."""
        self._default_apps = {'sim' : None, 'gen': None}
        Register.default_registry = self

    def register_calc(self, full_path, calc_type='sim', desc=None):
        """Registers a user application to libEnsemble

        Parameters
        ----------

        full_path: String
            The full path of the user application to be registered.

        calc_type: String
            Calculation type: Is this application part of a 'sim'
            or 'gen' function.

        desc: String, optional
            Description of this application.

        """
        rassert(calc_type in self._default_apps,
                "Unrecognized calculation type", calc_type)
        rassert(self._default_apps[calc_type] is None,
                "Default {} app already set".format(calc_type))
        self._default_apps[calc_type] = Application(full_path, calc_type, desc)


class BalsamRegister(Register):

    """Registers and stores user applications in libEnsemble and Balsam"""

    @staticmethod
    def del_apps():
        """Deletes all Balsam apps whose names contains .simfunc or .genfunc"""
        from balsam.service import models
        AppDef = models.ApplicationDefinition

        #Some error handling on deletes.... is it internal
        for app_type in ['.simfunc', '.genfunc']:
            deletion_objs = AppDef.objects.filter(name__contains=app_type)
            if deletion_objs:
                for del_app in deletion_objs.iterator():
                    logger.debug("Deleting app {}".format(del_app.name))
                deletion_objs.delete()

    @staticmethod
    def del_jobs():
        """Deletes all Balsam jobs whose names contains .simfunc or .genfunc"""
        from balsam.service import models
        Job = models.BalsamJob

        for app_type in ['.simfunc', '.genfunc']:
            deletion_objs = Job.objects.filter(name__contains=app_type)
            if deletion_objs:
                for del_job in deletion_objs.iterator():
                    logger.debug("Deleting job {}".format(del_job.name))
                deletion_objs.delete()

        ##May be able to use union function - to combine - see queryset help.
        ##Eg (not tested)
        #del_simfuncs = Job.objects.filter(name__contains='.simfunc')
        #del_genfuncs = Job.objects.filter(name__contains='.genfunc')
        #deletion_objs = deletion_objs.union()

    @staticmethod
    def add_app(name, exepath, desc):
        """ Add application to Balsam database """
        from balsam.service import models
        AppDef = models.ApplicationDefinition
        app = AppDef()
        app.name = name
        app.executable = exepath
        app.description = desc
        #app.default_preprocess = '' # optional
        #app.default_postprocess = '' # optional
        app.save()
        logger.debug("Added App {}".format(app.name))

    def __init__(self):
        """Instantiate a new BalsamRegister instance"""
        super().__init__()
        #Check for empty database if poss
        #And/or compare with whats in database and only empty if I need to
        #Currently not deleting as will delete the top level job -
        #   ie. the one running.
        #Will put MPI_MODE in a settings module...
        if MPI.COMM_WORLD.Get_rank() == 0:
            BalsamRegister.del_apps()
            BalsamRegister.del_jobs()


    def register_calc(self, full_path, calc_type='sim', desc=None):
        """Registers a user applications to libEnsemble and Balsam

        Parameters
        ----------

        full_path: String
            The full path of the user application to be registered.

        calc_type: String
            Calculation type: Is this application part of a 'sim'
            or 'gen' function.

        desc: String, optional
            Description of this application.

        """
        # OK to use Python 3 syntax (Balsam requires 3.6+)
        super().register_calc(full_path, calc_type, desc)

        rassert(calc_type in self._default_apps,
                "Unrecognized calculation type", calc_type)

        #Get from one place - so always matches
        calc_name = self._default_apps[calc_type].name
        desc = self._default_apps[calc_type].desc

        if MPI.COMM_WORLD.Get_rank() == 0:
            self.add_app(calc_name, full_path, desc)
