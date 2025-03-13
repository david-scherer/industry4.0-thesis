import os

from scheduler.entities.instance_mock_cpu import InstanceMock as InstanceMockCpu
from scheduler.entities.instance_mock import InstanceMock
from scheduler.entities.instance_base import InstanceBase
from scheduler.entities.serverless_funcion import Function
from scheduler.training.environment_LCAC import ServerlessSchedulingEnv
from scheduler.training.environment import ServerlessSchedulingEnv as ServerlessSchedulingEnvLCA

from ray.rllib.core import DEFAULT_MODULE_ID

import torch
from ray.rllib.core.rl_module.rl_module import RLModule

import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import logging
import datetime as dt
from scheduler.q_table.e1_learning import QLearning


rl_module: RLModule = RLModule.from_checkpoint(
        os.path.join(
            "/home/davestar/master-thesis/master-thesis/scheduler/models/LCAC_3",
            "learner_group",
            "learner",
            "rl_module",
            DEFAULT_MODULE_ID,
        )
    )

instances_dqn = [InstanceMockCpu('edge', 30, 0.99, (5, 15), randomized=False),
             InstanceMockCpu('private-cloud', 30, 0.99, (16, 25), randomized=False),
             InstanceMockCpu('public-cloud', 30, 0.99, (25, 40), randomized=False)]


functions = [Function(['critical'], 'test1'), 
             Function(['critical'], 'test2'), 
             Function(['critical'], 'coldstart')]

             
env = ServerlessSchedulingEnv(env_config={"providers": instances_dqn, "functions": functions, "simulate_cpu_inc_rates": False})

rt_dqn_list = []
success_dqn_list = []

cpu_load_edge = []
cpu_load_private_cloud = []
cpu_load_public_cloud = []

def action_dqn(obs, rl_module: RLModule, func: Function, instances: list[InstanceBase]):
    fwd_ins = {"obs": torch.Tensor([obs])}
    fwd_outputs = rl_module.forward_inference(fwd_ins)
    print(f"obs {obs} resulteed in action: {fwd_outputs}")
    action = fwd_outputs['actions'].item()
    
    rt, success = instances[action].call_function()
    if success:
        return rt, success
    else:
        return rt, success

# quantitative experiment, 3 strategies, 1000 iterations, 3 instances, 3 functions
# shows that cold starts are effectively avoided by the DQN agent
if __name__ == '__main__':
    instances_dqn[1].set_cpu_load(96)
    for inst in instances_dqn:
        inst.availability = 0.99
    for i in range(25):
        chosen_function = random.choice(functions)  
        obs = env.get_updated_state(instances=instances_dqn, func=chosen_function)

        rt_dqn, success_dqn = action_dqn(obs, rl_module, func=chosen_function, instances=instances_dqn)

        if success_dqn:
            rt_dqn_list.append(rt_dqn)

        for inst in instances_dqn:
            if inst.get_id() == 'edge':
                cpu_load_edge.append(inst.get_cpu_load())
            elif inst.get_id() == 'private-cloud':
                cpu_load_private_cloud.append(inst.get_cpu_load())
            elif inst.get_id() == 'public-cloud':
                cpu_load_public_cloud.append(inst.get_cpu_load())

        success_dqn_list.append(success_dqn)
        #success_dqn_lca_list.append(success_dqn_lca)


    # Plotting the CPU loads
    plt.figure(figsize=(10, 6))

    plt.plot(cpu_load_edge, label='edge')
    plt.plot(cpu_load_private_cloud, label='private-cloud')
    plt.plot(cpu_load_public_cloud, label='public-cloud')

    plt.xlabel('Time Step')
    plt.ylabel('CPU Load')
    plt.title('CPU Load per Instance')
    plt.legend()
    plt.grid(True)

    plt.savefig("e5.png")

    mean_rt_dqn = np.mean(rt_dqn_list)

    print(f"Mean Response Time (DQN): {mean_rt_dqn:.2f} ms")

    # Calculate and print failure rates
    failure_rate_dqn = 1 - np.mean(success_dqn_list)

    print(f"Failure Rate (DQN): {failure_rate_dqn:.2%}")
