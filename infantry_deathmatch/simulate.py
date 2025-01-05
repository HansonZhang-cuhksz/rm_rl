import gym
import numpy as np
from utils import *
from InfantryDeathmatch import *
from stable_baselines3 import PPO
import time

policy = {
    "shoot_reward": 10,
    "hit_reward": 100,
    "win_reward": 1000,
    "death_penalty": 0
}

# Initialize the environment
env = InfantryDeathmatch(policy, render=True)

model = PPO.load("ppo_robot_game_env", device='cpu')

done = False
env.reset()
counter = 0
while not done:
    # Infer action1 using the pre-trained model
    raw_obs = env.state[0]  # Get the observation for the second agent
    rotated_obs = rotate_obs(raw_obs)
    obs = np.concatenate([np.array(o).flatten() for o in rotated_obs])
    raw_action1, _ = model.predict(obs)
    action1 = rotate_action(raw_action1)
        
    # Infer action2 using the pre-trained model
    raw_obs = env.state[1]  # Get the observation for the second agent
    rotated_obs = rotate_obs(raw_obs)
    obs = np.concatenate([np.array(o).flatten() for o in rotated_obs])
    raw_action2, _ = model.predict(obs)
    action2 = rotate_action(raw_action2)

    obs, reward, done, info = env.step(action1, action2)
    output_obs = np.concatenate([np.array(o).flatten() for o in obs[0]])

    # Render the game
    env.render()

    time.sleep(0.05)
    counter += 1
    print("Counter: ", counter, "Player1 HP: ", env.player1.hp, "Player2 HP: ", env.player2.hp)

env.close()