import torch
from ray.rllib.core.rl_module.rl_module import RLModule
from scheduler.training.environment import ServerlessSchedulingEnv
from scheduler.entities.instance import Instance
from typing import List


def choose_instance(instances: List[Instance], float_values: List[float], rl_module: RLModule, env: ServerlessSchedulingEnv):
    obs = env.get_updated_state(instances=instances)

    fwd_ins = {"obs": torch.Tensor([obs])}
    fwd_outputs = rl_module.forward_inference(fwd_ins)
    
    action = fwd_outputs['actions'].item()
    #print(f"obs {obs} resulteed in action: {action}",flush=True)

    return action