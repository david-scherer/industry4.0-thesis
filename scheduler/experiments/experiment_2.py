import os

from scheduler.entities.instance_cs_real import InstanceMock
from scheduler.entities.serverless_funcion import Function
from scheduler.utils.config_reader import ConfigReader
from scheduler.training.environment import ServerlessSchedulingEnv

from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core import DEFAULT_MODULE_ID

import torch
from ray.rllib.core.rl_module.rl_module import RLModule
from scheduler.training.environment import ServerlessSchedulingEnv
from scheduler.entities.instance import Instance
from typing import List

import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

config = ConfigReader()
model_path = config.read_config("General", "MODEL_PATH")
rl_module: RLModule = RLModule.from_checkpoint(
        os.path.join(
            model_path,
            "learner_group",
            "learner",
            "rl_module",
            DEFAULT_MODULE_ID,
        )
    )

instances = [InstanceMock('edge', 30, 1, (5, 15)),
             InstanceMock('private-cloud', 30, 1, (5, 15)),
             InstanceMock('public-cloud', 30, 1, (5, 15))]
functions = [Function(['critical'], 'test1'), 
             Function(['critical'], 'test2'), 
             Function(['critical'], 'coldstart')]
env = ServerlessSchedulingEnv(env_config={"providers": instances, "functions": functions})

rt_heur_list = []
rt_dqn_list = []
rt_edge_list = []

success_heur_list = []
success_dqn_list = []
success_edge_list = []


def action_dqn(rl_module: RLModule, func: Function):
    obs = env.get_updated_state(instances=instances, func=func)
    fwd_ins = {"obs": torch.Tensor([obs])}
    fwd_outputs = rl_module.forward_inference(fwd_ins)
    print(f"obs {obs} resulteed in action: {fwd_outputs}")
    action = fwd_outputs['actions'].item()
    
    rt, success = instances[action].call_function(func)
    if success:
        return rt, success
    else:
        return rt, success


# greedy heuristic
def action_heur(func: Function):
    min_latency = 90000
    action = 0

    for index, ins in enumerate(instances):
        lat = ins.calc_weighted_latency_mean()
        if lat < min_latency:
            min_latency = lat
            action = index

    rt, success = instances[action].call_function(func)
    if success:
        return rt, success
    else:
        return rt, success

# 3 batches of calls, each with 15 requests, same function
# shows that cold starts are effectively avoided by the DQN agent
if __name__ == '__main__':
    for i in range(3):
        for i in range(15):
            func_heur = functions[0] 
            func_dqn = functions[1] 

            rt_heur, success_heur = action_heur(func_heur)
            rt_dqn, success_dqn = action_dqn(rl_module, func=func_dqn)


            if success_heur: 
                rt_heur_list.append(rt_heur)
            if success_dqn:
                rt_dqn_list.append(rt_dqn)


            success_heur_list.append(success_heur)
            success_dqn_list.append(success_dqn)
        for i in instances:
            i.reset()

    plt.figure(figsize=(10, 6))

    plt.plot(rt_dqn_list, label = 'DQN', color='blue')
    plt.plot(rt_heur_list, label = 'Heuristic', color='green')
    plt.xlabel('Requests')
    plt.ylabel('Response time')
    plt.title('Performance of different scheduling algorithms')
    plt.legend()
    plt.savefig("e2.png")

    mean_rt_heur = np.mean(rt_heur_list)
    mean_rt_dqn = np.mean(rt_dqn_list)

    print(f"Mean Response Time (Heuristic): {mean_rt_heur:.2f} ms")
    print(f"Mean Response Time (DQN): {mean_rt_dqn:.2f} ms")

    # Calculate and print failure rates
    failure_rate_heur = 1 - np.mean(success_heur_list)
    failure_rate_dqn = 1 - np.mean(success_dqn_list)

    print(f"Failure Rate (Heuristic): {failure_rate_heur:.2%}")
    print(f"Failure Rate (DQN): {failure_rate_dqn:.2%}")
