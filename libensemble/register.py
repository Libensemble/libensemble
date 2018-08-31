#!/usr/bin/env python

"""Module to register applications to libEnsemble"""

import os
import logging
from mpi4py import MPI

logger = logging.getLogger(__name__)
#For debug messages in this module  - uncomment (see libE.py to change root logging level)
#logger.setLevel(logging.DEBUG)

class RegistrationException(Exception): pass


class Application:

    '''An application is an executable user-program (e.g. Implementing a sim/gen function).'''

    def __init__(self, full_path, calc_type='sim', desc=None, default=True):
        '''Instantiate a new Application instance.'''
        self.full_path = full_path
        self.calc_type = calc_type
        self.default = default
        self.calc_dir, self.exe = os.path.split(full_path)

        #Dont change:Why? - cos will use this name to delete jobs in database - see del_apps(), del_jobs()
        self.name = self.exe + '.' + self.calc_type + 'func'
        self.desc = desc or (self.exe + ' ' + self.calc_type + ' function')

#May merge this into job_controller
class Register():

    '''Registers and stores user applications

    Attributes
    ----------
    default_registry : Obj: Register or inherited class.
        A class attribute holding the default registry.

    '''

    default_registry = None

    @property
    def sim_default_app(self):
        """Return the default simulation app."""
        return self._default_apps['sim']

    @property
    def gen_default_app(self):
        """Return the default generator app."""
        return self._default_apps['gen']

    def __init__(self, default=True):
        '''Instantiate a new Register instance

        Parameters
        ----------

        default: Boolean, optional
            Make this the default_registry (Default is True)


        Note: Currently, only a default registry is supported.

        '''
        self._default_apps = {'sim' : None, 'gen': None}
        if default:
            Register.default_registry = self

    def register_calc(self, full_path, calc_type='sim', desc=None, default=True):
        '''Registers a user application to libEnsemble

        Parameters
        ----------

        full_path: String
            The full path of the user application to be registered.

        calc_type: String
            Calculation type: Is this application part of a 'sim' or 'gen' function.

        desc: String, optional
            Description of this application.

        default: Boolean, optional
            Register to the default_registry (Default is True).


        '''
        if not default:
            return # Always default currently
        if calc_type not in self._default_apps:
            raise RegistrationException("Unrecognized calculation type", calc_type)
        if self._default_apps[calc_type] is not None:
            raise RegistrationException("Default {} app already set".format(calc_type))
        self._default_apps[calc_type] = Application(full_path, calc_type, desc, default)


class BalsamRegister(Register):

    '''Registers and stores user applications in libEnsemble and Balsam'''

    @staticmethod
    def del_apps():
        """ Deletes all Balsam apps whose names contains substring .simfunc or .genfunc"""
        import balsam.launcher.dag
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
        """ Deletes all Balsam jobs whose names contains substring .simfunc or .genfunc"""
        import balsam.launcher.dag
        from balsam.service import models
        Job = models.BalsamJob

        for app_type in ['.simfunc', '.genfunc']:
            deletion_objs = Job.objects.filter(name__contains=app_type)
            if deletion_objs:
                for del_job in deletion_objs.iterator():
                    logger.debug("Deleting job {}".format(del_job.name))
                deletion_objs.delete()

        ##May be able to use union function - to combine - see queryset help. Eg (not tested)
        #del_simfuncs = Job.objects.filter(name__contains='.simfunc')
        #del_genfuncs = Job.objects.filter(name__contains='.genfunc')
        #deletion_objs = deletion_objs.union()

    @staticmethod
    def add_app(name, exepath, desc):
        """ Add application to Balsam database """
        import balsam.launcher.dag
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

    def __init__(self, default=True):
        '''Instantiate a new BalsamRegister instance

        Parameters
        ----------

        default: Boolean, optional
            Make this the default_registry (Default is True)


        Note: Currently, only a default registry is supported.

        '''

        super().__init__(default)
        #Check for empty database if poss
        #And/or compare with whats in database and only empty if I need to

        #Currently not deleting as will delete the top level job - ie. the one running.

        #Will put MPI_MODE in a settings module...
        if MPI.COMM_WORLD.Get_rank() == 0:
            BalsamRegister.del_apps()
            BalsamRegister.del_jobs()


    def register_calc(self, full_path, calc_type='sim', desc=None, default=True):
        '''Registers a user applications to libEnsemble and Balsam

        Parameters
        ----------

        full_path: String
            The full path of the user application to be registered.

        calc_type: String
            Calculation type: Is this application part of a 'sim' or 'gen' function.

        desc: String, optional
            Description of this application.

        default: Boolean, optional
            Register to the default_registry (Default is True).


        '''
        super().register_calc(full_path, calc_type, desc, default)
        #Req python 3 to exclude args - but as Balsam requires 3.6+ I may do - or is it only __init__()

        #Get from one place - so always matches
        if calc_type in self._default_apps:
            calc_name = self._default_apps[calc_type].name
            desc = self._default_apps[calc_type].desc
        else:
            raise RegistrationException("Unrecognized calculation type", calc_type)

        if MPI.COMM_WORLD.Get_rank() == 0:
            self.add_app(calc_name, full_path, desc)
