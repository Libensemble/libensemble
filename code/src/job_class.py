import time
from message_numbers import EVAL_SIM_TAG, EVAL_GEN_TAG

class Job():
        
    def __init__(self):
        self.time = 0.0
        self.start = 0.0
        self.end = 0.0
        self.pid = 0 #process ID - not currently used.
        self.calc_type = None
        #self.jobid = 
        
    def start_timer(self):
        self.start = time.time()
    
    
    def stop_timer(self):
        self.end = time.time()
        #increment so can start and stop repeatedly
        self.time += self.end - self.start
        
#    def get_time():
#        return self.time
    
    #Should use message_numbers - except i want to separate type for being just a tag.
    def get_type(self):
        if self.calc_type==EVAL_SIM_TAG:
            return 'sim'
        elif self.calc_type==EVAL_GEN_TAG:
            return 'gen' 
        elif self.calc_type==None:
            return 'No type set'
        else:
            return 'Unknown type'
