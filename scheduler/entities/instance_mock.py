from scheduler.entities.serverless_funcion import Function
from scheduler.entities.instance_base import InstanceBase
import random
import time

class InstanceMock(InstanceBase):
    def __init__(self, *args, **kwargs):  # Add cpu_load parameter
            super().__init__(*args, **kwargs) 
            self.cs = 0

    def calc_cold_start(self, func: Function):
        cs_prob = 0.1
        if func.get_name() == 'coldstart':
            cs_prob = 0.7
        if random.random() < cs_prob:
            cs = True
        else:
            cs = False
        self.latest_cs = cs
        return cs
    
    def call_function(self):
        response_time = 0
        if random.random() > self.availability:
            return 1000, False  
        
        shape = 2 
        scale = self.latency_range[1] / shape 
        response_time = random.gammavariate(shape, scale)
                
        if random.random() > 0.95:
            response_time += random.randint(15, 40)

        #if random.random() > 0.95 and self.spike_start_time is None:
        #    self.spike_start_time = time.time()
        #    self.spike_duration = random.randint(2, 4)
        #    self.spike_magnitude = random.randint(40, 60)

        #if self.spike_start_time is not None:
        #    if time.time() - self.spike_start_time < self.spike_duration:
        #        response_time += self.spike_magnitude
        #    else:
        #        self.spike_start_time = None

        self.add_latency(response_time)
        # add additional computation time
        response_time += 5

        if self.latest_cs:
            cold_start_delay = random.gauss(400, 100)  # Mean = 500ms, Std Dev = 100ms
            cold_start_delay = max(0, cold_start_delay)  # Ensure delay is not negative
            response_time += cold_start_delay

        return response_time, True
    
    def reset(self):
        self.function_name_dict = {}
        self.spike_start_time = None

    def set_cold_start(self, cs):
        self.cs = cs

    def get_cold_start(self):
        return self.cs
