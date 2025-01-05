import gym
import numpy as np
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from gym import spaces
import math

from RobotGameEnv import RobotGameEnv

class SingleAgentRobotGameEnv(gym.Env):
    def __init__(self, policy):
        super().__init__()
        self.original_env = RobotGameEnv(policy, render=False)
        
        # Flatten the action space
        self.action_space = spaces.Box(
            low=np.array([-1, -1, 0, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Flatten the observation space
        low = np.concatenate([space.low.flatten() for space in self.original_env.observation_space.spaces])
        high = np.concatenate([space.high.flatten() for space in self.original_env.observation_space.spaces])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self):
        obs = self.original_env.reset()[0]
        return np.concatenate([np.array(o).flatten() for o in obs])

    def step(self, action):
        move = action[0:2]
        attack = 1 if action[2] >= 0.5 else 0
        attack_direction = action[3:5]
        formatted_action = (move, attack, attack_direction)

        # Default (do-nothing) action for agent2
        action2 = (
            np.array([0.0, 0.0], dtype=np.float32),  # move
            0,  # attack=0
            np.array([0.0, 0.0], dtype=np.float32)   # direction
        )

        env_out = self.original_env.step(formatted_action, action2)
        obs, reward, done, info = env_out
        this_obs = obs[0]
        formatted_obs = [this_obs[0][0], this_obs[0][1]], [this_obs[1][0], this_obs[1][1]], this_obs[2], this_obs[3], this_obs[4], this_obs[5], this_obs[6]
        output_obs = np.concatenate([np.array(o).flatten() for o in formatted_obs])
        return output_obs, reward[0], done, info
    
    def render(self, mode='human'):
        return self.original_env.render(mode)
    
class SwitchAgentRobotGameEnv(gym.Env):
    def __init__(self, model_pth, policy):
        super().__init__()
        self.original_env = RobotGameEnv(policy, render=False)

        # Load the pre-trained model for the second agent
        self.model = PPO.load(model_pth, device='cpu')
        
        # Flatten the action space
        self.action_space = spaces.Box(
            low=np.array([-1, -1, 0, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )
        
        # Flatten the observation space
        low = np.concatenate([space.low.flatten() for space in self.original_env.observation_space.spaces])
        high = np.concatenate([space.high.flatten() for space in self.original_env.observation_space.spaces])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self):
        obs = self.original_env.reset()[0]
        return np.concatenate([np.array(o).flatten() for o in obs])

    def step(self, action):
        move = action[0:2]
        attack = 1 if action[2] >= 0.5 else 0
        attack_direction = action[3:5]
        formatted_action = (move, attack, attack_direction)

        # Infer action2 using the pre-trained model
        raw_obs = self.original_env.state[1]  # Get the observation for the second agent
        obs = np.concatenate([np.array(o).flatten() for o in raw_obs])
        action2, _ = self.model.predict(obs)
        move2 = action2[0:2]
        attack2 = 1 if action2[2] >= 0.5 else 0
        attack_direction2 = action2[3:5]
        formatted_action2 = (move2, attack2, attack_direction2)

        env_out = self.original_env.step(formatted_action, formatted_action2)
        obs, reward, done, info = env_out
        this_obs = obs[0]
        formatted_obs = [this_obs[0][0], this_obs[0][1]], [this_obs[1][0], this_obs[1][1]], this_obs[2], this_obs[3], this_obs[4], this_obs[5], this_obs[6]
        output_obs = np.concatenate([np.array(o).flatten() for o in formatted_obs])
        return output_obs, reward[0], done, info

class CustomCallback(BaseCallback):
    def __init__(self, check_freq, policy, verbose=1):
        super(CustomCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.policy = policy

    def _on_step(self) -> bool:
        training_env = self.training_env.envs[0].env.gym_env

        if self.policy["shoot_reward"] > 1:
            self.policy["shoot_reward"] /= self.n_calls + 1
        if self.n_calls > 200000:
            if self.policy["hit_reward"] > 5:
                self.policy["hit_reward"] /= self.n_calls / 200000
            self.policy["death_penalty"] = 800
        if self.n_calls > 400000:
            self.policy["shoot_reward"] = -1

        training_env.original_env.policy = self.policy

        if self.n_calls % self.check_freq == 0:
            print(f"Step: {self.n_calls}, Time elapsed: {time.time() - start_time:.2f} seconds")
        return True

import time

start_time = time.time()

policy = {
    "shoot_reward": 10,
    "hit_reward": 100,
    "win_reward": 1000,
    "death_penalty": 0
}

# Step Single Agent
env = SingleAgentRobotGameEnv(policy)
model = PPO("MlpPolicy", env, verbose=0, device='cpu')
model.learn(total_timesteps=10000, callback=CustomCallback(check_freq=2000, policy=policy))
model.save("ppo_robot_game_env")
print("--- %s seconds ---" % (time.time() - start_time))

# Step Switch Agent
env = SwitchAgentRobotGameEnv("ppo_robot_game_env", policy)
model = PPO.load("ppo_robot_game_env", env=env, verbose=0, device='cpu')
model.learn(total_timesteps=1000000, callback=CustomCallback(check_freq=2000, policy=policy))
model.save("ppo_robot_game_env")
print("--- %s seconds ---" % (time.time() - start_time))

# Step With Block