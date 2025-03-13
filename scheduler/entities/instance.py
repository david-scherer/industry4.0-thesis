from scheduler.entities.serverless_funcion import Function
from scheduler.entities.instance_base import InstanceBase
import datetime as dt
import random

class Instance(InstanceBase):
    def calc_cold_start(self, func: Function):
        if func.name in self.function_name_dict:
            print(f'Function exists in {self.function_name_dict}')
            diff = dt.datetime.now() - self.function_name_dict[func.name] 
            if diff.total_seconds() >= self.threshold:
                return 0
            else:
                return 1
        else: 
            print(f'Not found in {self.function_name_dict}')
            self.update_cold_start_dict(func)
            return 0
    
    def call_function(self):
        response_time = 0
        if random.random() > self.availability:
            return 1, False  
        
        response_time = random.randint(*self.latency_range)
        
        if random.random() > 0.95:
            response_time += random.randint(15, 40)

        self.add_latency(response_time)
        # add additional computation time
        response_time += 5

        if self.latest_cs:
            response_time += 500

        return response_time, True
    
    def reset(self):
        self.function_name_dict = {}

