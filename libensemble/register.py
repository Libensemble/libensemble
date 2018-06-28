#!/usr/bin/env python

"""Module to register applications to libEnsemble."""

import os
import subprocess
import logging
from mpi4py import MPI

logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(name)s (%(levelname)s): %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

#For debug messages - uncomment
logger.setLevel(logging.DEBUG)

class RegistrationException(Exception): pass


class Application:
    
    '''An application is an executable user-program (e.g. Implementing a sim/gen function).'''

    def __init__(self, full_path, calc_type='sim', desc=None, default=True):
        '''Instantiate a new Application instance.'''
        self.full_path = full_path
        self.calc_type = calc_type
        self.desc = desc
        self.default = default
        self.calc_dir, self.exe = os.path.split(full_path)
        
        #Dont change:Why? - cos will use this name to delete jobs in database - see del_apps(), del_jobs()
        self.name = self.exe + '.' + self.calc_type + 'func'
        
        if desc is None:
            self.desc = self.exe + ' ' + self.calc_type + ' function'

#May merge this into job_controller
class Register():
    
    '''Registers and stores user applications'''
    
    default_registry = None
    
    def __init__(self, default=True):
        '''Instantiate a new Register instance.'''
        self.sim_default_app = None
        self.gen_default_app = None        
        if default:
            Register.default_registry = self

        #logger.debug("default_registry is {}".format(Register.default_registry))
        #logger.debug("default_registry sim name is {}".format(Register.default_registry.sim_default_app))
            
    def register_calc(self, full_path, calc_type='sim', desc=None, default=True):
        '''Registers a user applications to libEnsemble.'''
        if default:
            if calc_type == 'sim':
                if self.sim_default_app is not None:
                    #logger - either error or overwrite - error
                    raise RegistrationException("Default sim app already set")
                else:
                    #Register.sim_default_app = full_path
                    app = Application(full_path, calc_type, desc, default)
                    self.sim_default_app = app
            elif calc_type == 'gen':
                if self.gen_default_app is not None:
                    #logger - either error or overwrite - error
                    raise RegistrationException("Default gen app already set")
                else:
                    #Register.sim_default_app = full_path
                    app = Application(full_path, calc_type, desc, default)
                    self.gen_default_app = app
            else:
                #Raise Exception **
                raise RegistrationException("Unrecognized calculation type", calc_type)
        else:
            pass #always default currently
        
        #return id(app)
        return
 
 
class BalsamRegister(Register):
    
    '''Registers and stores user applications in libEnsemble and Balsam'''
    
    @staticmethod
    def del_apps():
        """ Deletes all Balsam apps whose names contains substring .simfunc or .genfunc"""         
        import balsam.launcher.dag as dag
        from balsam.service import models
        AppDef = models.ApplicationDefinition 
        #deletion_objs = AppDef.objects.all()
        
        #Some error handling on deletes.... is it internal
        deletion_objs = AppDef.objects.filter(name__contains='.simfunc')
        if deletion_objs:
            for del_app in deletion_objs.iterator():
                logger.debug("Deleting app {}".format(del_app.name))
            deletion_objs.delete()
        deletion_objs = AppDef.objects.filter(name__contains='.genfunc')
        if deletion_objs:
            for del_app in deletion_objs.iterator():
                logger.debug("Deleting app {}".format(del_app.name))
            deletion_objs.delete()           

    @staticmethod
    def del_jobs():
        """ Deletes all Balsam jobs whose names contains substring .simfunc or .genfunc"""
        import balsam.launcher.dag as dag
        from balsam.service import models
        Job = models.BalsamJob
        #deletion_objs = Job.objects.all()
        
        deletion_objs = Job.objects.filter(name__contains='.simfunc')
        if deletion_objs:
            for del_job in deletion_objs.iterator():
                logger.debug("Deleting job {}".format(del_job.name))
            deletion_objs.delete()
        deletion_objs = Job.objects.filter(name__contains='.genfunc')
        if deletion_objs:
            for del_job in deletion_objs.iterator():
                logger.debug("Deleting job {}".format(del_job.name))
            deletion_objs.delete()        
        
        ##May be able to use union function - to combine - see queryset help. Eg (not tested)
        #del_simfuncs = Job.objects.filter(name__contains='.simfunc')
        #del_genfuncs = Job.objects.filter(name__contains='.genfunc')     
        #deletion_objs = deletion_objs.union()
        
    @staticmethod       
    def add_app(name,exepath,desc):
        """ Add application to Balsam database """
        import balsam.launcher.dag as dag
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
        '''Instantiate a new BalsamRegister instance.'''
        super().__init__()
        #Check for empty database if poss
        #And/or compare with whats in database and only empty if I need to
        
        #Currently not deleting as will delete the top level job - ie. the one running.
        
        #Will put MPI_MODE in a settings module...
        if MPI.COMM_WORLD.Get_rank() == 0:
            BalsamRegister.del_apps()
            BalsamRegister.del_jobs()
        
    
    def register_calc(self, full_path, calc_type='sim', desc=None, default=True):
        '''Registers a user applications to libEnsemble and Balsam.'''
        super().register_calc(full_path, calc_type, desc, default) 
        #Req python 3 to exclude args - but as Balsam requires 3.6+ I may do - or is it only __init__()
        
        #calc_dir, calc_name = os.path.split(full_path)    

        #Get from one place - so always matches
        if calc_type == 'sim':
            calc_name = self.sim_default_app.name
            desc = self.sim_default_app.desc
        elif calc_type == 'gen':
            calc_name = self.gen_default_app.name
            desc = self.gen_default_app.desc
        else:
            #Raise Exception **
            raise RegistrationException("Unrecognized calculation type", calc_type)
        
        #if desc is None:
            #desc = calc_exe + ' ' + calc_type + ' function'
        if MPI.COMM_WORLD.Get_rank() == 0:
            self.add_app(calc_name, full_path, desc)

