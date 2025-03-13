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
import pickle

import logging
import datetime as dt
from scheduler.q_table.e1_learning import QLearning


rl_module: RLModule = RLModule.from_checkpoint(
        os.path.join(
            "/home/davestar/master-thesis/master-thesis/scheduler/models/LCAC_2",
            "learner_group",
            "learner",
            "rl_module",
            DEFAULT_MODULE_ID,
        )
    )

rl_module_LCA: RLModule = RLModule.from_checkpoint(
        os.path.join(
            "/home/davestar/master-thesis/master-thesis/scheduler/models/LCA_3",
            "learner_group",
            "learner",
            "rl_module",
            DEFAULT_MODULE_ID,
        )
    )

instances_heur = [InstanceMockCpu('edge', 30, 0.99, (5, 15)),
             InstanceMockCpu('private-cloud', 30, 0.99, (16, 25)),
             InstanceMockCpu('public-cloud', 30, 0.99, (25, 40))]
instances_dqn = [InstanceMockCpu('edge', 30, 0.99, (5, 15), randomized=False),
             InstanceMockCpu('private-cloud', 30, 0.99, (16, 25), randomized=False),
             InstanceMockCpu('public-cloud', 30, 0.99, (25, 40), randomized=False)]
instances_dqn_lca = [InstanceMockCpu('edge', 30, 0.99, (5, 15)),
             InstanceMockCpu('private-cloud', 30, 0.99, (16, 25)),
             InstanceMockCpu('public-cloud', 30, 0.99, (25, 40))]

functions = [Function(['critical'], 'test1'), 
             Function(['critical'], 'test2'), 
             Function(['critical'], 'coldstart')]

logging.basicConfig(filename=f'/tmp/learning-logs_{dt.datetime.now().strftime("%m-%d %H:%M:%S")}',
                    filemode='a',
                    format='%(asctime)s, %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)

logger = logging.getLogger('learning-logs')

#q_learning = QLearning(instances_qtable, logger=logger)
#q_learning.e1_init_or_load_soph_qtable()

#q_learning.e1_train_soph(experiment=True)
             
env = ServerlessSchedulingEnv(env_config={"providers": instances_dqn, "functions": functions, "simulate_cpu_inc_rates": False})
env_LCA = ServerlessSchedulingEnvLCA(env_config={"providers": instances_dqn, "functions": functions, "simulate_cpu_inc_rates": False})

#with open('/home/davestar/master-thesis/master-thesis/e1-qtable.pickle', "rb") as f:
#    q_table = pickle.load(f)
#with open('/home/davestar/master-thesis/master-thesis/e1-qcounter.pickle', "rb") as f:
#    q_counter = pickle.load(f)

rt_heur_list = []
rt_dqn_list = []
rt_dqn_lca_list = []

success_heur_list = []
success_dqn_list = []
success_dqn_lca_list = []


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
    

#def action_qtable(func: Function):
#    obs = tuple(((int(inst.calc_weighted_latency_mean()/20) * 20), inst.calc_cold_start(func=func)) for inst in instances_qtable)
#    action = np.argmin(q_table[obs])
#
#    print(f"obs {obs} resulted in action: {action}")
#
#    rt, success = instances_qtable[action].call_function()
#    if success:
#        return rt, success
#    else:
#        return rt, success

# greedy heuristic
def action_heur(func: Function):
    min_latency = 90000
    action = 0

    for index, ins in enumerate(instances_heur):
        lat = ins.calc_weighted_latency_mean()
        if lat < min_latency:
            min_latency = lat
            action = index

    instances_heur[action].calc_cold_start(func=func)
    rt, success = instances_heur[action].call_function()
    if success:
        return rt, success
    else:
        return rt, success

# quantitative experiment, 3 strategies, 1000 iterations, 3 instances, 3 functions
# shows that cold starts are effectively avoided by the DQN agent
if __name__ == '__main__':
    for i in range(80):
        chosen_function = random.choice(functions)  
        obs = env.get_updated_state(instances=instances_dqn, func=chosen_function)
        obs_LCA = env_LCA.get_updated_state(instances=instances_dqn_lca, func=chosen_function)

        rt_heur, success_heur = action_heur(chosen_function)
        rt_dqn, success_dqn = action_dqn(obs, rl_module, func=chosen_function, instances=instances_dqn)
        #rt_dqn_lca, success_dqn_lca = action_dqn(obs_LCA, rl_module_LCA, func=chosen_function, instances=instances_dqn_lca)

        if success_heur: 
            rt_heur_list.append(rt_heur)
        if success_dqn:
            rt_dqn_list.append(rt_dqn)
        #if success_dqn_lca:
        #    rt_dqn_lca_list.append(rt_dqn_lca)

        success_heur_list.append(success_heur)
        success_dqn_list.append(success_dqn)
        #success_dqn_lca_list.append(success_dqn_lca)

    plt.figure(figsize=(10, 6))

    # Combine the response time data into a list of lists
    data = [rt_heur_list, rt_dqn_list, rt_dqn_lca_list] 

    # Create the box plot using seaborn
    sns.violinplot(data=data)

    # Set x-axis labels
    plt.xticks(ticks=[0, 1, 2], labels=["Heuristic", "DQN", "DQN_LCA"])  

    plt.xlabel("Strategy")
    plt.ylabel("Response Time (ms)")
    plt.title("Response Time Comparison")
    plt.savefig("e4.png")

    mean_rt_heur = np.mean(rt_heur_list)
    mean_rt_dqn = np.mean(rt_dqn_list)
    mean_rt_dqn_lca = np.mean(rt_dqn_lca_list)
    #mean_rt_qtable = np.mean(rt_qtable_list)

    print(f"Mean Response Time (Heuristic): {mean_rt_heur:.2f} ms")
    print(f"Mean Response Time (DQN): {mean_rt_dqn:.2f} ms")
    print(f"Mean Response Time (DQN LCA): {mean_rt_dqn_lca:.2f} ms")
    #print(f"Mean Response Time (Q-table): {mean_rt_qtable:.2f} ms")

    # Calculate and print failure rates
    failure_rate_heur = 1 - np.mean(success_heur_list)
    failure_rate_dqn = 1 - np.mean(success_dqn_list)
    failure_rate_dqn_lca = 1 - np.mean(success_dqn_lca_list)
    #failure_rate_qtable = 1 - np.mean(success_qtable_list)

    print(f"Failure Rate (Heuristic): {failure_rate_heur:.2%}")
    print(f"Failure Rate (DQN): {failure_rate_dqn:.2%}")
    print(f"Failure Rate (DQN LCA): {failure_rate_dqn_lca:.2%}")
    #print(f"Failure Rate (Q-table): {failure_rate_qtable:.2%}")