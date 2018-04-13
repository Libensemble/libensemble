#!/usr/bin/env python

""" Script to set up apps for balsam """

#Quit possibly make this part of job/job_controller module 
#An app object could be an attribute of job - so job relates to this app

import os
import subprocess
import logging

logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(name)s (%(levelname)s): %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

#For debug messages - uncomment
logger.setLevel(logging.DEBUG)

class RegistrationException(Exception): pass


class Application:
    def __init__(self, full_path, calc_type='sim', desc=None, default=True):
        self.full_path = full_path
        self.calc_type = calc_type
        self.desc = desc
        self.default = default
        self.calc_dir, self.name = os.path.split(full_path)
        
        if desc is None:
            self.desc = self.name + ' ' + self.calc_type + ' function'

#Think I will merge this into job_controller
class Register():
    
    default_registry = None    
    
    def __init__(self, default=True):
        self.sim_default_app = None
        self.gen_default_app = None        
        if default:
            Register.default_registry = self

        logger.debug("default_registry is {}".format(Register.default_registry))
        #logger.debug("default_registry sim name is {}".format(Register.default_registry.sim_default_app))
            
    def register_calc(self, full_path, calc_type='sim', desc=None, default=True):
        
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
 
 
#Think I will merge this into job_controller
class BalsamRegister(Register):
 
    @staticmethod
    def del_apps():
        """ For now just deletes all apps """
        import balsam.launcher.dag as dag
        from balsam.service import models
        AppDef = models.ApplicationDefinition 
        deletion_objs = AppDef.objects.all()
        deletion_objs.delete()
        logger.debug("deleted apps")

    @staticmethod
    def del_jobs():
        """ For now just deletes all jobs """
        import balsam.launcher.dag as dag
        from balsam.service import models
        Job = models.BalsamJob
        deletion_objs = Job.objects.all()
        deletion_objs.delete()
        logger.debug("deleted jobs")  
        
    @staticmethod       
    def add_app(name,exepath,desc):
        """ Add application to database """
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
        super().__init__()
        #Check for empty database if poss
        #And/or compare with whats in database and only empty if I need to
        BalsamRegister.del_apps()
        BalsamRegister.del_jobs()
        
    
    def register_calc(self, full_path, calc_type='sim', desc=None, default=True):
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
            
        self.add_app(calc_name, full_path, desc)
