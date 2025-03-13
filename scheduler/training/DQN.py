import ray
import os
from ray.rllib.algorithms.dqn import DQNConfig
from environment import ServerlessSchedulingEnv
import torch
import matplotlib.pyplot as plt
import pickle

from scheduler.entities.instance_mock import InstanceMock
from scheduler.entities.serverless_funcion import Function

from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.rl_module.rl_module import RLModule

ray.init()
instances = []
instances.append(InstanceMock('edge', 30, 0.95, (2, 15)))
instances.append(InstanceMock('private-cloud', 30, 0.99, (16, 25)))
instances.append(InstanceMock('public-cloud', 30, 0.9995, (25, 40)))

functions = [Function(['critical'], 'test1'), 
             Function(['critical'], 'test2'), 
             Function(['critical'], 'coldstart')]

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
                    "functions": functions},
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
r_min_list = []
loss_list = []

print("\nStarting training iterations...\n")
for i in range(100):
    result = dqn_agent.train()

    # Extract relevant information
    time_this_iter = result["time_this_iter_s"]
    episode_reward_mean = result["env_runners"]["episode_return_mean"]  # Extract mean reward
    episode_reward_max = result["env_runners"]["episode_return_max"]
    episode_reward_min = result["env_runners"]["episode_return_min"]
    learner_loss = result["learners"]["default_policy"]["total_loss"]

    r_mean_list.append(episode_reward_mean)
    r_max_list.append(episode_reward_max)
    loss_list.append(learner_loss)
    r_min_list.append(episode_reward_min)

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
plt.plot(r_min_list, label='Min Reward')

plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('DQN Convergence')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("rewards.png")

plt.figure(figsize=(10, 6))
plt.plot(r_mean_list, label='Mean Reward')

plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('DQN Convergence')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("mean-rewards.png")
    
with open("reward_loss_data.pkl", "wb") as file:
    pickle.dump((r_mean_list, r_max_list, r_min_list, loss_list), file)

print("Data saved successfully.")

print('\n\nsaving now...')
checkpoint_dir = dqn_agent.save(checkpoint_dir="/home/davestar/master-thesis/master-thesis/scheduler/models/LCA_4,6")
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
action_dist_class = rl_module.get_inference_action_dist_cls()
for _ in range(100):
    fwd_ins = {"obs": torch.Tensor([obs])}
    fwd_outputs = rl_module.forward_inference(fwd_ins)
    print(f"obs {obs} resulteed in action: {fwd_outputs}")
    # this can be either deterministic or stochastic distribution
    action_dist = action_dist_class.from_logits(
        fwd_outputs["actions"]
    )
    #action = action_dist.sample()[0].numpy()
    obs, reward, terminated, truncated, info = env.step(fwd_outputs['actions'].item())


    #input_dict = {Columns.OBS: torch.from_numpy(obs).unsqueeze(0)}
    #rl_module_out = rl_module.forward_inference(input_dict)["actions"]
    #action_index = rl_module_out.numpy()
    #env.step(action_index[0])
    #print(rl_module_out)
    #print(action_index)

print("\n MOIS THIS WAS FOR REAL NOW\n")
print(checkpoint_dir.checkpoint.path)
with open("./resultpath.txt", "w") as f:
    f.write(checkpoint_dir.checkpoint.path)