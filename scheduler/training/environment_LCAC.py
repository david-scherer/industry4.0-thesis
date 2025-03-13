from gymnasium.spaces import Discrete, Box
import gymnasium as gym
import numpy as np
from typing import Optional
from scheduler.entities.instance_mock_cpu import InstanceMock as Instance
from scheduler.entities.serverless_funcion import Function
import random
from typing import List

class ServerlessSchedulingEnv(gym.Env):
    def __init__(self, env_config: Optional[dict] = None):
        config = env_config or {}
        self.avilability_bins = [0, 90, 95, 98, 99, 99.5, 99.9, 100]
        self.instances: list[Instance] = config['providers']
        self.functions: list[Function] = config['functions']
        self.num_providers = len(self.instances)

        # indicates the simulation of only 1 instance overload faster than others
        # this helps explore more states and patterns
        if 'simulate_cpu_inc_rates' in config:
            self.simulate_cpu_inc_rates = config['simulate_cpu_inc_rates']
        else:
            self.simulate_cpu_inc_rates = True
        
        self.action_space = Discrete(self.num_providers) 
        # Define the state space
        # Assuming you have 4 features per provider (latency, cold start, availability)
        # to add 1 function label, you need to adjust the shape to (self.num_providers * 3 + 1)
        self.observation_space = Box(
            low=-1, high=40, shape=(self.num_providers * 4,), dtype=np.float32
        )
        self.current_state = self.reset()

    def reset(self, seed=None, options={}):
        self.counter = 0
        
        for i in self.instances:
            self.set_random_availability(i)
            #i.set_randomized(random.choice([True, False]))
            i.reset()

        if self.simulate_cpu_inc_rates:
            single = random.choice([True, False])
            cpu_rate_inst = random.randint(0, 2)
            for i, inst in enumerate(self.instances):
                if cpu_rate_inst == i:
                    if single:
                        inst.set_cpu_incr_rate(1.0)
                    else :
                        inst.set_cpu_incr_rate(random.uniform(0, 0.3))
                else:
                    if single:
                        inst.set_cpu_incr_rate(random.uniform(0, 0.3))
                    else:
                        inst.set_cpu_incr_rate(1.0)
        return self.get_updated_state(instances=self.instances), {}

    # for training
    def step(self, action):
        # Take the given action (choose a provider)
        chosen_instance: Instance = self.instances[action]

        response_time, success = chosen_instance.call_function()

        # Calculate the reward based on the outcome
        reward = self._calculate_reward(response_time=response_time, success=success, cpu_load=chosen_instance.get_cpu_load())

        chosen_function = random.choice(self.functions)    

        #print(f"{self.current_state} -> reward: {reward} from action: {action}, success: {success}, response time: {response_time}, cpu load: {chosen_instance.get_cpu_load()}")

        # Update the environment state
        self.current_state = self.get_updated_state(instances=self.instances, func=chosen_function)

        #for instance in self.instances:
        #    instance.update_cpu_load()

        # Check if the episode is done
        self.counter += 1
        done = self.counter == 80     
        # Return the next state, reward, done flag, and any extra info

        return self.current_state, reward, done, False, {}

    def _calculate_reward(self, response_time, success, cpu_load): 
        reward = 0
        if not success:
            reward -= 40

        if cpu_load >= 95:   
            reward -= 60
        elif cpu_load >= 90:
            reward -= 40
        elif cpu_load >= 80:
            reward -= 20
            
        reward -= (response_time) / 10
        return reward
    
    def test_reward(self, response_time, success, cpu_load):

        # 1. Baseline Reward (encourage successful execution)
        baseline_reward = 1 if success else -1  

        # 2. CPU Utilization Penalty (same penalty as failure for high CPU load)
        cpu_threshold = 90  # Example: Threshold for high CPU load (in percentage)
        cpu_penalty = -1 if cpu_load > cpu_threshold else 0
        cpu_penalty -= 0.8 * (cpu_load / 100)
        

        # 3. Response Time Penalty (discourage high response times / cold starts)
        max_response_time = 1.0  # Example: Maximum acceptable response time of 1 second
        response_time_penalty = -0.6 * (response_time / max_response_time)  

        # --- Combine Reward Components ---
        reward = baseline_reward + cpu_penalty + response_time_penalty

        return reward

        
    # for each instance:
    def get_updated_state(self, instances: List[Instance], func: Function = Function([], '')):
        state = []
        for instance in instances:
            state.extend([
                self._normalize_latency(instance),
                instance.calc_cold_start(func),  
                self._normalize_availability(instance=instance),
                self._normalize_cpu_load(instance)
            ])
        # state.append(self.normalize_function_label(func.get_first_label)) 
        return np.array(state, dtype=np.float32)
    
    def _normalize_cpu_load(self, instance: Instance):
        cpu_load = instance.get_cpu_load()
        return int(cpu_load / 10) 
    
    def _normalize_latency(self, instance: Instance):
        if instance.calc_weighted_latency_mean() > 400:
            return 40
        return int(np.floor(instance.calc_weighted_latency_mean() / 10))
    
    def _normalize_availability(self, instance: Instance):
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
        
    def set_random_availability(self, instance: Instance):
        weights = [0.05, 0.05, 0.15, 0.2, 0.2, 0.15, 0.1, 0.1] 

        random_value = random.choices( self.avilability_bins, weights=weights)[0]  # Use choices() with weights
        instance.set_availability(random_value / 100)

    def get_instances(self):
        return self.instances