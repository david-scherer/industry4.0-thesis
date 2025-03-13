from scheduler.entities.serverless_funcion import Function
import datetime as dt
import random
from abc import ABC, abstractmethod
import copy
import json

class InstanceBase(ABC):
    def __init__(self, id: str, threshold: int, availability: float,  latency_range: tuple):
        self.id = id
        self.threshold = threshold
        self.availability = availability
        self.latency_range = latency_range
        self.spike_start_time = None
        self.cpu_load = 0.2

        # state parameters
        self.function_name_dict = {}
        self.latencies = []
        self.latencies.append(random.randint(*latency_range))
        self.latencies.append(random.randint(*latency_range))
        self.latencies.append(random.randint(*latency_range))

    def get_id(self):
        return self.id

    def get_availability(self):
        return self.availability
    
    def set_availability(self, availability: float):
        self.availability = availability

    def add_latency(self, l):
        if len(self.latencies) >= 3:
            self.latencies.pop(0)
            self.latencies.append(l)
        else:
            self.latencies.append(l)
    
    def calc_weighted_latency_mean(self):
        return (self.latencies[0]*0.1 + self.latencies[1]*0.3 + self.latencies[2]*0.6)

    def update_cold_start_dict(self, func: Function):
        self.function_name_dict[func.name] = dt.datetime.now()
    
    @abstractmethod
    def call_function(self):
        pass
    
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def calc_cold_start(self, func: Function):
        pass

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


    def __str__(self):
        return json.dumps({
            "id": self.id,
            "threshold": self.threshold,
            "availability": self.availability,
            "latency_range": self.latency_range,
            "function_name_dict": {
                func_name: timestamp.isoformat()
                for func_name, timestamp in self.function_name_dict.items()
            },
            "latencies": self.latencies,
        })
    
    def deepcopy(self):
        return copy.deepcopy(self)