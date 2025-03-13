from scheduler.entities.serverless_funcion import Function
from scheduler.entities.instance_base import InstanceBase
import random
import time

class InstanceMock(InstanceBase):
    def __init__(self, *args, randomized=False, cpu_incr_rate=1.0, **kwargs):  # Add cpu_load parameter
        super().__init__(*args, **kwargs) 
        self.load_change_counter = 0
        self.randomized = randomized
        self.cpu_load = 10
        self.cpu_incr_rate = cpu_incr_rate


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
    
    def decrease_cpu_load(self, amount):
        self.cpu_load -= amount
        self.cpu_load = max(0, self.cpu_load) 

    def get_cpu_load(self):
        #if random.random() > 0.99:
        #    return random.randint(0, 100)
        if self.randomized:
            return random.randint(0, 100)
        else:
            return self.cpu_load
    
    def set_cpu_load(self, load):
        self.cpu_load = load
    
    def update_cpu_load(self):
        self.load_change_counter += 1

        # Adjust load every N function calls (example: every 10 calls)
        if self.load_change_counter >= 10:
            self.load_change_counter = 0

            # Choose a load change pattern (with a chance for decrease)
            pattern = random.choices(
                ["gradual", "spike", "idle", "decrease"], 
                weights=[0.3, 0.2, 0.2, 0.3]  # Adjust weights as needed
            )[0]

            if pattern == "gradual":
                self.cpu_load += random.randint(-3, 3)  # Smaller change
            elif pattern == "spike":
                self.cpu_load += random.randint(5, 15)  # Smaller spike
            elif pattern == "idle":
                self.cpu_load -= random.randint(5, 15) 
            elif pattern == "decrease":  # More frequent decreases
                self.cpu_load -= random.randint(2, 8) 

            self.cpu_load = max(0, min(100, self.cpu_load))
    
    def call_function(self):
        response_time = 0
        if random.random() > self.availability:
            return 1, False  
        
        shape = 2 
        scale = self.latency_range[1] / shape 
        response_time = random.gammavariate(shape, scale)
                
        if random.random() > 0.95:
            response_time += random.randint(15, 40)

        # cloud spike simulation
        #if self.id == 'public-cloud' and random.random() > 0.95 and self.spike_start_time is None:
        #    self.spike_start_time = time.time()
        #    self.spike_duration = random.randint(1, 2)
        #    self.spike_magnitude = random.randint(30, 60)

        #if self.spike_start_time is not None:
        #    if time.time() - self.spike_start_time < self.spike_duration:
        #        response_time += self.spike_magnitude
        #    else:
        #        self.spike_start_time = None

        # CPU load simulation
        if random.random() <= self.cpu_incr_rate:
            cpu_usage = random.randint(3, 5) 
            self.cpu_load += cpu_usage
            self.cpu_load = min(self.cpu_load, 100)

        self.add_latency(response_time)
        # add additional computation time
        response_time += 5

        if self.latest_cs:
            cold_start_delay = random.gauss(400, 100)  # Mean = 500ms, Std Dev = 100ms
            cold_start_delay = max(0, cold_start_delay)  # Ensure delay is not negative
            response_time += cold_start_delay

        return response_time, True
    
    def set_randomized(self, randomized):
        self.randomized = randomized

    def set_cpu_incr_rate(self, cpu_incr_rate):
        self.cpu_incr_rate = cpu_incr_rate
    
    def reset(self):
        self.function_name_dict = {}
        self.cpu_load = random.randint(0, 15)

