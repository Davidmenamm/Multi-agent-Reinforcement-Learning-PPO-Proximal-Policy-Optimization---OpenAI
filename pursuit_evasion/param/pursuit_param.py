from stable_baselines3.ppo import CnnPolicy
from stable_baselines3 import PPO
from pettingzoo.butterfly import pursuit_v3  # Changed to pursuit evasion environment
import supersuit as ss
import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback

# Callback class for saving best reward
class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq, log_dir, verbose=1):
        super().__init__(verbose)
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

# Training function
def training(run):
    if run:
        # Main environment setup for pursuit evasion
        env = pursuit_v3.parallel_env()

        # Environment adaptations
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class="stable_baselines3")

        # PPO learning
        log_dir = "ppo_pursuit_evasion_logs"
        os.makedirs(log_dir, exist_ok=True)
        callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
        model = PPO(CnnPolicy, env, verbose=3, gamma=0.99, n_steps=512, ent_coef=0.01, learning_rate=0.0001, vf_coef=0.5, max_grad_norm=0.5, gae_lambda=0.95, n_epochs=10, clip_range=0.2, batch_size=64)
        model.learn(total_timesteps=2000000, callback=callback)
        model.save("policy_pursuit_evasion")

        # Plot training rewards
        rewards = [info['r'] for info in callback.model.ep_info_buffer if 'r' in info]
        plt.figure(figsize=(12, 8))
        plt.plot(rewards)
        plt.xlabel('Timestep')
        plt.ylabel('Rewards')
        plt.title('Training Rewards Over Time')
        plt.savefig(os.path.join(log_dir, 'training_rewards.png'))
        plt.show()

        env.close()

# Model evaluation function
def evaluate_model(env, model, num_episodes=10, render_every=20):
    for episode in range(num_episodes):
        obs = env.reset()
        episode_rewards = 0
        step = 0
        for agent in env.agent_iter():
            step += 1
            obs, reward, termination, truncation, info = env.last()
            act = model.predict(obs, deterministic=True)[0] if not termination else None
            env.step(act)
            episode_rewards += reward
            if step % render_every == 0:
                env.render()
            if termination or truncation:
                break
        print(f'Episode {episode + 1}: Total Reward: {episode_rewards}')

# Visualization function
def visualize(run):
    if run:
        # Load trained policy
        model = PPO.load("policy_pursuit_evasion")

        # Environment setup for evaluation
        env = pursuit_v3.env(render_mode="human")
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

        # Evaluate the model
        evaluate_model(env, model)

        env.close()

# Main function
def main():
    training(False)  # Set to True to train, False to skip training
    visualize(True)  # Set to True to visualize

# Run main
if __name__ == '__main__':
    main()
