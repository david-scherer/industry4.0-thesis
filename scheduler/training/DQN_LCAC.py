import ray
import os
from ray.rllib.algorithms.dqn import DQNConfig
from environment_LCAC import ServerlessSchedulingEnv
import torch

from scheduler.entities.instance_mock_cpu import InstanceMock
from scheduler.entities.serverless_funcion import Function

from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.algorithms.algorithm import Algorithm

LOAD_FROM_CHECKPOINT = True

ray.init()
instances = []
instances.append(InstanceMock('edge', 30, 0.95, (2, 15), randomized=True))
instances.append(InstanceMock('private-cloud', 30, 0.99, (16, 25), randomized=True))
instances.append(InstanceMock('public-cloud', 30, 0.9995, (25, 40), randomized=True))

functions = [Function(['critical'], 'test1'), 
             Function(['critical'], 'test2'), 
             Function(['critical'], 'coldstart')]

if LOAD_FROM_CHECKPOINT:
    print("loading from checkpoint...")
    checkpoint_dir = "/home/davestar/master-thesis/master-thesis/scheduler/models/LCAC_1"
    dqn_agent = Algorithm.from_checkpoint(checkpoint_dir)
    dqn_agent.config["env"] = ServerlessSchedulingEnv(env_config={"providers": instances, "functions": functions, "simulate_cpu_inc_rates": True})
else:
    config: DQNConfig = (
        DQNConfig()
        .env_runners(num_env_runners=3)
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True
        )
        .environment(
            ServerlessSchedulingEnv,
            env_config={"providers": instances,
                        "functions": functions,
                        "simulate_cpu_inc_rates": False},
        )
        .framework("torch")
        .training(replay_buffer_config={
            "type": "PrioritizedEpisodeReplayBuffer",
            "capacity": 60000,
            "alpha": 0.5,
            "beta": 0.5,
        })
    )
    dqn_agent = config.build()

print("\nStarting training iterations...\n")
for i in range(500):
    result = dqn_agent.train()

    if i % 100 == 0:
        print(dqn_agent.evaluate())
        print()
    # Extract relevant information
    time_this_iter = result["time_this_iter_s"]
    episode_reward_mean = result["env_runners"]
    learner_loss = result["learners"]["default_policy"]["total_loss"]

    # Print the extracted information
    print(f"Time of iteration nr {i}: {time_this_iter:.2f} seconds")
    print(f"Mean episode reward: {episode_reward_mean}")
    print(f"Learner loss: {learner_loss}\n")

        

print('\n\nsaving now...')
checkpoint_dir = dqn_agent.save(checkpoint_dir="/home/davestar/master-thesis/master-thesis/scheduler/models/LCAC_3")
print('...to this location: \n')
print(checkpoint_dir.checkpoint.path)
print(dqn_agent.evaluate())
dqn_agent.stop()

rl_module: RLModule = RLModule.from_checkpoint(
        os.path.join(
            checkpoint_dir.checkpoint.path,
            "learner_group",
            "learner",
            "rl_module",
            DEFAULT_MODULE_ID,
        )
    )

env = ServerlessSchedulingEnv(env_config={"providers": instances, "functions": functions})

obs, info = env.reset()
for inst in instances:
    inst.availability = 0.95
reward = None
action_dist_class = rl_module.get_inference_action_dist_cls()
for i in range(200):
    fwd_ins = {"obs": torch.Tensor([obs])}
    fwd_outputs = rl_module.forward_inference(fwd_ins)
    # this can be either deterministic or stochastic distribution
    action_dist = action_dist_class.from_logits(
        fwd_outputs["actions"]
    )
    if reward is not None:
        print(f"{i} Obs {obs} resulted in action: {fwd_outputs} with reward: {reward}")
    #action = action_dist.sample()[0].numpy()
    obs, reward, terminated, truncated, info = env.step(fwd_outputs['actions'].item())
    

print(checkpoint_dir.checkpoint.path)
with open("./resultpath.txt", "w") as f:
    f.write(checkpoint_dir.checkpoint.path)