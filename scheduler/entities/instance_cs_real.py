from scheduler.entities.serverless_funcion import Function
from scheduler.entities.instance_base import InstanceBase
import random
import time
import datetime as dt

class InstanceMock(InstanceBase):
   
    def call_function(self, func: Function):
        response_time = 0
        if random.random() > self.availability:
            return 1, False

        

        shape = 2 
        scale = self.latency_range[1] / shape 
        response_time = random.gammavariate(shape, scale)
                
        if random.random() > 0.95:
            response_time += random.randint(15, 40)

        if self.id == 'public-cloud' and random.random() > 0.95 and self.spike_start_time is None:
            self.spike_start_time = time.time()
            self.spike_duration = random.randint(2, 4)
            self.spike_magnitude = random.randint(30, 60)

        if self.spike_start_time is not None:
            if time.time() - self.spike_start_time < self.spike_duration:
                response_time += self.spike_magnitude
            else:
                self.spike_start_time = None

        self.add_latency(response_time)
        # add additional computation time
        response_time += 5

        if self.calc_cold_start(func):
            cold_start_delay = random.gauss(400, 100)  # Mean = 500ms, Std Dev = 100ms
            cold_start_delay = max(0, cold_start_delay)  # Ensure delay is not negative
            response_time += cold_start_delay

        self.update_cold_start_dict(func)

        return response_time, True
    
    def reset(self):
        self.function_name_dict = {}

    def calc_cold_start(self, func: Function):
        if func.name in self.function_name_dict:
            print(f'Function exists in {self.function_name_dict}')
            diff = dt.datetime.now() - self.function_name_dict[func.name] 
            if diff.total_seconds() >= self.threshold:
                return True
            else:
                return False
        else: 
            print(f'Not found in {self.function_name_dict}')
            #self.update_cold_start_dict(func)
            return True

