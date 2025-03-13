import ray
import os
from ray.rllib.algorithms.dqn import DQNConfig
from environment_LCAL import ServerlessSchedulingEnv
import torch
import matplotlib.pyplot as plt

from scheduler.entities.instance_mock import InstanceMock
from scheduler.entities.serverless_funcion import Function

from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.algorithms.algorithm import Algorithm

LOAD_FROM_CHECKPOINT = False

ray.init()
instances = []
instances.append(InstanceMock('edge', 30, 0.95, (2, 15)))
instances.append(InstanceMock('private-cloud', 30, 0.99, (16, 25)))
instances.append(InstanceMock('public-cloud', 30, 0.9995, (25, 40)))

functions = [Function(['success-critical'], 'test1'), 
             Function(['response-time-critical'], 'test2'), 
             Function([''], 'coldstart')]

if LOAD_FROM_CHECKPOINT:
    print("\nloading from checkpoint...\n")
    checkpoint_dir = "/home/davestar/master-thesis/master-thesis/scheduler/models/LCAL_4"
    dqn_agent = Algorithm.from_checkpoint(checkpoint_dir)
    dqn_agent.config["env"] = ServerlessSchedulingEnv(env_config={"providers": instances, "functions": functions, "training": True, "label": "success-critical"})
else:
    config: DQNConfig = (
        DQNConfig()
        .env_runners(num_env_runners=4)
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True
        )
        .environment(
            ServerlessSchedulingEnv,
            env_config={"providers": instances,
                        "functions": functions,
                        "training": True},
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


r_mean_list = []
r_max_list = []
loss_list = []

print("\nStarting training iterations...\n")
for i in range(300):
    result = dqn_agent.train()
    # Extract relevant information
    time_this_iter = result["time_this_iter_s"]
    episode_reward_mean = result["env_runners"]["episode_return_mean"]  # Extract mean reward
    episode_reward_max = result["env_runners"]["episode_return_max"]
    learner_loss = result["learners"]["default_policy"]["total_loss"]

    r_mean_list.append(episode_reward_mean)
    r_max_list.append(episode_reward_max)
    loss_list.append(learner_loss)

    # Print the extracted information
    print(f"Time of iteration nr {i}: {time_this_iter:.2f} seconds")
    print(f"Learner loss: {learner_loss}\n")

dqn_agent.config["env"] = ServerlessSchedulingEnv(env_config={"providers": instances, "functions": functions, "training": True, "label": "response-time-critical"})
for i in range(300):
    result = dqn_agent.train()
    # Extract relevant information
    time_this_iter = result["time_this_iter_s"]
    episode_reward_mean = result["env_runners"]["episode_return_mean"]  # Extract mean reward
    episode_reward_max = result["env_runners"]["episode_return_max"]
    learner_loss = result["learners"]["default_policy"]["total_loss"]

    r_mean_list.append(episode_reward_mean)
    r_max_list.append(episode_reward_max)
    loss_list.append(learner_loss)

    # Print the extracted information
    print(f"Time of iteration nr {i}: {time_this_iter:.2f} seconds")
    print(f"Learner loss: {learner_loss}\n")

# Plotting the convergence graph
plt.figure(figsize=(10, 6))
plt.plot(loss_list, label='Loss')

plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('DQN Convergence')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("loss.png")

# Plotting the convergence graph
plt.figure(figsize=(10, 6))
plt.plot(loss_list[1:], label='Loss')

plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('DQN Convergence')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("lossv2.png")

plt.figure(figsize=(10, 6))
plt.plot(r_mean_list, label='Mean Reward')
plt.plot(r_max_list, label='Max Reward')

plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('DQN Convergence')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("rewards.png")

print('\n\nsaving now...')
checkpoint_dir = dqn_agent.save(checkpoint_dir="/home/davestar/master-thesis/master-thesis/scheduler/models/LCAL_1")
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