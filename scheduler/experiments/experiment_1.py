import os

from scheduler.entities.instance_mock import InstanceMock
from scheduler.entities.serverless_funcion import Function
from scheduler.utils.config_reader import ConfigReader

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

instances = [InstanceMock('edge', 30, 0.99, (5, 15)),
             InstanceMock('private-cloud', 30, 0.99, (16, 25)),
             InstanceMock('public-cloud', 30, 0.99, (25, 40))]
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


def action_dqn(obs, rl_module: RLModule, func: Function):
    fwd_ins = {"obs": torch.Tensor([obs])}
    fwd_outputs = rl_module.forward_inference(fwd_ins)
    print(f"obs {obs} resulteed in action: {fwd_outputs}")
    action = fwd_outputs['actions'].item()
    
    rt, success = instances[action].call_function()
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

    rt, success = instances[action].call_function()
    if success:
        return rt, success
    else:
        return rt, success

# quantitative experiment, 3 strategies, 1000 iterations, 3 instances, 3 functions
# shows that cold starts are effectively avoided by the DQN agent
if __name__ == '__main__':
    for i in range(1000):
        chosen_function = random.choice(functions)  
        obs = env.get_updated_state(instances=instances, func=chosen_function)

        rt_heur, success_heur = action_heur(chosen_function)
        rt_dqn, success_dqn = action_dqn(obs, rl_module, func=chosen_function)
        rt_edge, success_edge = instances[0].call_function()

        if success_heur: 
            rt_heur_list.append(rt_heur)
        if success_dqn:
            rt_dqn_list.append(rt_dqn)
        if success_edge:
            rt_edge_list.append(rt_edge)

        success_heur_list.append(success_heur)
        success_dqn_list.append(success_dqn)
        success_edge_list.append(success_edge)

    plt.figure(figsize=(10, 6))

    # Combine the response time data into a list of lists
    data = [rt_heur_list, rt_dqn_list, rt_edge_list] 

    # Create the box plot using seaborn
    sns.violinplot(data=data)

    # Set x-axis labels
    plt.xticks(ticks=[0, 1, 2], labels=["Heuristic", "DQN", "Edge"])  

    plt.xlabel("Strategy")
    plt.ylabel("Response Time (ms)")
    plt.title("Response Time Comparison")
    plt.savefig("e1.png")

    mean_rt_heur = np.mean(rt_heur_list)
    mean_rt_dqn = np.mean(rt_dqn_list)
    mean_rt_edge = np.mean(rt_edge_list)

    print(f"Mean Response Time (Heuristic): {mean_rt_heur:.2f} ms")
    print(f"Mean Response Time (DQN): {mean_rt_dqn:.2f} ms")
    print(f"Mean Response Time (Edge): {mean_rt_edge:.2f} ms")

    # Calculate and print failure rates
    failure_rate_heur = 1 - np.mean(success_heur_list)
    failure_rate_dqn = 1 - np.mean(success_dqn_list)
    failure_rate_edge = 1 - np.mean(success_edge_list)

    print(f"Failure Rate (Heuristic): {failure_rate_heur:.2%}")
    print(f"Failure Rate (DQN): {failure_rate_dqn:.2%}")
    print(f"Failure Rate (Edge): {failure_rate_edge:.2%}")