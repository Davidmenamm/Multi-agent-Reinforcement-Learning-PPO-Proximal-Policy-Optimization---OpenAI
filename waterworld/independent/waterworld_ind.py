# code in process..

from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
from pettingzoo.sisl import waterworld_v3
import supersuit as ss
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

# Callback class definition
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    A custom callback that saves the model when it achieves a new best reward.
    """
    def __init__(self, check_freq, log_dir, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self):
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            rewards = [info['r'] for info in self.model.ep_info_buffer if 'r' in info]
            if len(rewards) > 0:
                mean_reward = np.mean(rewards)
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    self.model.save(self.save_path)
        return True

# Initialize and train independent models
def train_independent_models(env, num_agents, log_dir):
    models = []
    for agent_idx in range(num_agents):
        agent_log_dir = os.path.join(log_dir, f'agent_{agent_idx}')
        os.makedirs(agent_log_dir, exist_ok=True)
        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=agent_log_dir)
        
        model = PPO(CnnPolicy, env, verbose=3, gamma=0.99, n_steps=512, ent_coef=0.01, learning_rate=0.0001, vf_coef=0.5, max_grad_norm=0.5, gae_lambda=0.95, n_epochs=10, clip_range=0.2, batch_size=64)
        model.learn(total_timesteps=2000000 // num_agents, callback=callback)
        models.append(model)
        model.save(os.path.join(agent_log_dir, "policy"))
    return models

# Training function
def training(run):
    if run:
        env = waterworld_v3.parallel_env(n_pursuers=10, n_evaders=50, n_poison=50, n_food=50, obstacle_radius=0.04, food_reward=10, poison_penalty=-1, encounter_reward=0.01, n_coop=2, sensor_range=0.2, max_cycles=500)
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class="stable_baselines3")

        num_agents = 10  # Number of agents
        models = train_independent_models(env, num_agents, "ppo_waterworld_logs")

        env.close()

# Evaluation function for independent models
def evaluate_independent_models(env, models, num_episodes=10, render_every=20):
    for episode in range(num_episodes):
        obs = env.reset()
        episode_rewards = [0 for _ in models]
        for agent in env.agent_iter():
            agent_idx = int(agent.split('_')[-1])  # Extract agent index
            action = models[agent_idx].predict(obs[agent], deterministic=True)[0]
            obs, rewards, dones, infos = env.step(action)
            episode_rewards[agent_idx] += rewards[agent]
            if all(dones.values()):
                break
        print(f'Episode {episode + 1}: Total Reward: {sum(episode_rewards)}')

# Visualization function
def visualize(run):
    if run:
        models = [PPO.load(f"ppo_waterworld_logs/agent_{i}/policy") for i in range(10)]  # Load models for each agent

        env = waterworld_v3.env(n_pursuers=10, n_evaders=50, n_poison=50, n_food=50, obstacle_radius=0.04, food_reward=10, poison_penalty=-1, encounter_reward = 1, render_mode="human")

        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

        evaluate_independent_models(env, models)

        env.close()

# Main function
def main():
    training(False)  # Set to True to train models
    visualize(True)  # Visualize the trained models

if __name__ == '__main__':
    main()
