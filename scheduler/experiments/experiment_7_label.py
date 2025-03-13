import os

from scheduler.entities.instance_mock import InstanceMock
from scheduler.entities.instance_base import InstanceBase
from scheduler.entities.serverless_funcion import Function
from scheduler.training.environment_LCAL import ServerlessSchedulingEnv

from ray.rllib.core import DEFAULT_MODULE_ID

import torch
from ray.rllib.core.rl_module.rl_module import RLModule

import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

rl_module_LCAL: RLModule = RLModule.from_checkpoint(
        os.path.join(
            "/home/davestar/master-thesis/master-thesis/scheduler/models/LCAL_1",
            "learner_group",
            "learner",
            "rl_module",
            DEFAULT_MODULE_ID,
        )
    )

instances_dqn_lcal = [InstanceMock('edge', 30, 0.99, (5, 15)),
             InstanceMock('private-cloud', 30, 0.70, (16, 25)),
             InstanceMock('public-cloud', 30, 0.70, (25, 40))]

functions = [Function(['success-critical'], 'test'), 
             Function(['response-time-critical'], 'coldstart')]

labels = ["success-critical", "response-time-critical"]

env_LCAL = ServerlessSchedulingEnv(env_config={"providers": instances_dqn_lcal, "functions": functions})

rt_dqn_lcal_list = []

success_dqn_lcal_list = []

def action_dqn(obs, rl_module: RLModule, func: Function, instances: list[InstanceBase]):
    fwd_ins = {"obs": torch.Tensor([obs])}
    fwd_outputs = rl_module.forward_inference(fwd_ins)
    print(f"obs {obs} resulteed in action: {fwd_outputs} with func label {func.get_first_label()}")
    action = fwd_outputs['actions'].item()
    
    rt, success = instances[action].call_function()
    if success:
        return rt, success
    else:
        return rt, success


if __name__ == '__main__':
    for f_ct in range(2):
        curr_rt = []
        curr_success = []
        for f in functions:
            f.set_label_experimental(labels[f_ct])
        for i in range(1000):
            #chosen_function = random.choice(functions)  
            chosen_function = random.choice(functions)

            obs_LCAL = env_LCAL.get_updated_state(instances=instances_dqn_lcal, func=chosen_function)

            rt_dqn_lcal, success_dqn_lcal = action_dqn(obs_LCAL, rl_module_LCAL, func=chosen_function, instances=instances_dqn_lcal)

            if success_dqn_lcal: 
                curr_rt.append(rt_dqn_lcal)
            curr_success.append(success_dqn_lcal)
        rt_dqn_lcal_list.append(curr_rt)
        success_dqn_lcal_list.append(curr_success)


    plt.figure(figsize=(10, 6))

    # Combine the response time data into a list of lists
    data = [rt_dqn_lcal_list[0], rt_dqn_lcal_list[1]] 

    # Create the box plot using seaborn
    sns.violinplot(data=data)

    # Set x-axis labels
    plt.xticks(ticks=[0, 1, 2], labels=["Success-critical", "Default", "Response-time-critical"])  

    plt.xlabel("Label")
    plt.ylabel("Response Time (ms)")
    plt.title("Response Time Comparison")
    plt.savefig("e7_label.png")

    mean_rt_dqn_lca_sc = np.mean(rt_dqn_lcal_list[0])
    mean_rt_dqn_lca_d = np.mean(rt_dqn_lcal_list[1])

    print(f"Mean Response Time sc: {mean_rt_dqn_lca_sc:.2f} ms")
    print(f"Mean Response Time default: {mean_rt_dqn_lca_d:.2f} ms")

    # Calculate and print failure rates
    failure_rate_dqn_lcal_sc = 1 - np.mean(success_dqn_lcal_list[0])
    failure_rate_dqn_lcal_d = 1 - np.mean(success_dqn_lcal_list[1])

    print(f"Failure Rate sc: {failure_rate_dqn_lcal_sc:.2%}")
    print(f"Failure Rate default: {failure_rate_dqn_lcal_d:.2%}")
