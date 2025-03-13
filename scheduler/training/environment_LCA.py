from gymnasium.spaces import Discrete, Box
import gymnasium as gym
import numpy as np
from typing import Optional
from scheduler.entities.instance_base import InstanceBase
from scheduler.entities.serverless_funcion import Function
import random
from typing import List


class ServerlessSchedulingEnv(gym.Env):
    def __init__(self, env_config: Optional[dict] = None):
        config = env_config or {}
        self.avilability_bins = [0, 90, 95, 98, 99, 99.5, 99.9, 100]
        self.instances: list[InstanceBase] = config['providers']
        self.functions: list[Function] = config['functions']
        self.num_providers = len(self.instances)
        
        self.action_space = Discrete(self.num_providers) 
        # Define the state space
        # Assuming you have 4 features per provider (latency, cold start, availability)
        # to add 1 function label, you need to adjust the shape to (self.num_providers * 3 + 1)
        self.observation_space = Box(
            low=-1, high=40, shape=(self.num_providers * 3,), dtype=np.float32
        )
        self.current_state = self.reset()

    def reset(self, seed=None, options={}):
        self.counter = 0
        #for i in self.instances:
        #    self.set_random_availability(i)
        #    i.reset()

        print("Resetting environment")
        return self.get_updated_state(instances=self.instances), {}

    # for training
    def step(self, action):
        # Take the given action (choose a provider)
        chosen_instance: InstanceBase = self.instances[action]

        response_time, success = chosen_instance.call_function()

        # Calculate the reward based on the outcome
        reward = self._calculate_reward(response_time=response_time, action=action, success=success)

        chosen_function = random.choice(self.functions)             
        
        # Update the environment state
        self.current_state = self.get_updated_state(instances=self.instances, func=chosen_function)

        # Check if the episode is done
        self.counter += 1
        done = self.counter == 50        
        # Return the next state, reward, done flag, and any extra info
        return self.current_state, reward, done, False, {}


    # using function labels, we can adjust the reward calculation to meet the requirements
    def _calculate_reward(self, response_time, action, success): 
        reward = 0
        #if action == 0: # edge
        #    reward = 0
        #elif action == 1: # private cloud
        #    reward = -0
        #elif action == 2: # public cloud
        #    reward = -0
        if not success:
            reward -= 40
            
        reward -= (response_time) / 10
        return reward
    
    # for each instance:
    def get_updated_state(self, instances: List[InstanceBase], func: Function = Function([], '')):
        state = []
        for instance in instances:
            state.extend([
                self._normalize_latency(instance),
                instance.calc_cold_start(func),  
                self._normalize_availability(instance=instance)
            ])
        # state.append(self.normalize_function_label(func.get_first_label)) 
        return np.array(state, dtype=np.float32)
    
    def _normalize_latency(self, instance: InstanceBase):
        if instance.calc_weighted_latency_mean() > 400:
            return 40
        return int(np.floor(instance.calc_weighted_latency_mean() / 10))
    
    def _normalize_availability(self, instance: InstanceBase):
        availability = instance.get_availability()
        return np.digitize(availability * 100,  self.avilability_bins) - 1

    # currently we support only 1 label per function
    # to allow more, we need to adjust our obs space box to allow more values
    def normalize_function_label(self, name):
        if name == 'test1':
            return 1
        elif name == 'test2':
            return 2
        elif name == 'coldstart':
            return 3
        else:
            return 0
        
    def set_random_availability(self, instance: InstanceBase):
        weights = [0.05, 0.05, 0.15, 0.2, 0.2, 0.15, 0.1, 0.1] 

        random_value = random.choices( self.avilability_bins, weights=weights)[0]  # Use choices() with weights
        instance.set_availability(random_value / 100)

    def get_instances(self):
        return self.instances