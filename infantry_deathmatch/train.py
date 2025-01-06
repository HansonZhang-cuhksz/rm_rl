import gym
from gym import spaces
import numpy as np
from stable_baselines3 import PPO, A2C, DDPG, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback

from InfantryDeathmatch import InfantryDeathmatch
from utils import rotate_action, rotate_obs
from aim_edge_agent import decide

model = None

class SingleAgentEnv(gym.Env):
    def __init__(self, policy):
        super().__init__()
        self.original_env = InfantryDeathmatch(policy, do_render=False)
        
        # Flatten the action space
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([1, 2*np.pi, 2*np.pi], dtype=np.float32),
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
        action1 = action
        # Second agent's action is None
        action2 = (0, 0, 5 * np.pi / 4)
        obs, reward, done, info = self.original_env.step(action1, action2)
        output_obs = np.concatenate([np.array(o).flatten() for o in obs[0]])
        return output_obs, reward[0], done, info
    
class AimEdgeAgentEnv(gym.Env):
    def __init__(self, policy):
        super().__init__()
        self.original_env = InfantryDeathmatch(policy, do_render=False)
        
        # Flatten the action space
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([1, 2*np.pi, 2*np.pi], dtype=np.float32),
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
        action1 = action
        
        raw_obs = self.original_env.state[1]  # Get the observation for the second agent
        rotated_obs = rotate_obs(raw_obs)
        obs = np.concatenate([np.array(o).flatten() for o in rotated_obs])
        raw_action2 = decide(obs)
        action2 = rotate_action(raw_action2)

        obs, reward, done, info = self.original_env.step(action1, action2)
        output_obs = np.concatenate([np.array(o).flatten() for o in obs[0]])
        return output_obs, reward[0], done, info
    
class SwitchAgentEnv(gym.Env):
    def __init__(self, model_pth, policy):
        super().__init__()
        self.original_env = InfantryDeathmatch(policy, do_render=False)

        # Load the pre-trained model for the second agent
        self.model = PPO.load(model_pth, device='cpu')
        
        # Flatten the action space
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([1, 2*np.pi, 2*np.pi], dtype=np.float32),
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
        action1 = action
        
        # Infer action2 using the pre-trained model
        raw_obs = self.original_env.state[1]  # Get the observation for the second agent
        rotated_obs = rotate_obs(raw_obs)
        obs = np.concatenate([np.array(o).flatten() for o in rotated_obs])
        raw_action2, _ = self.model.predict(obs)
        action2 = rotate_action(raw_action2)

        obs, reward, done, info = self.original_env.step(action1, action2)
        output_obs = np.concatenate([np.array(o).flatten() for o in obs[0]])
        return output_obs, reward[0], done, info

class CustomCallback(BaseCallback):
    def __init__(self, check_freq, policy, sync_save=False, verbose=1):
        super(CustomCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.policy = policy
        self.sync_save = sync_save

    def _on_step(self) -> bool:
        global model

        training_env = self.training_env.envs[0].env.gym_env
        training_env.original_env.policy = self.policy

        if self.n_calls % self.check_freq == 0:
            if self.sync_save:
                model.save("ppo_robot_game_env")
            print(f"Step: {self.n_calls}, Time elapsed: {time.time() - start_time:.2f} seconds")
        return True

import time

start_time = time.time()

policy = {
    "aim_reward_bias": 1,
    "aim_reward_weight": 10,
    "shoot_reward": 100,
    "move_reward": 1,
    "hit_wall_penalty": 0
}

# Step Single Agent
env = SingleAgentEnv(policy)
model = PPO("MlpPolicy", env, verbose=1, device='cpu')
# model = A2C("MlpPolicy", env, verbose=1, device='cpu')
model.learn(total_timesteps=4000, callback=CustomCallback(check_freq=10, policy=policy))
# model.learn(total_timesteps=100)
model.save("ppo_robot_game_env")
print("--- %s seconds ---" % (time.time() - start_time))

# Step Aim Edge Agent
env = AimEdgeAgentEnv(policy)
model = PPO.load("ppo_robot_game_env", env=env, verbose=1, device='cpu')
# model = PPO("MlpPolicy", env, verbose=1, device='cpu')
model.learn(total_timesteps=100000, callback=CustomCallback(check_freq=10, policy=policy))
model.save("ppo_robot_game_env")
print("--- %s seconds ---" % (time.time() - start_time))

# Step Switch Agent
env = SwitchAgentEnv("ppo_robot_game_env", policy)
model = PPO.load("ppo_robot_game_env", env=env, verbose=1, device='cpu')
# model = PPO("MlpPolicy", env, verbose=1, device='cpu')
model.learn(total_timesteps=1000000, callback=CustomCallback(check_freq=10, policy=policy, sync_save=True))
model.save("ppo_robot_game_env")
print("--- %s seconds ---" % (time.time() - start_time))