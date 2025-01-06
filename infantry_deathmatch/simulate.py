import gym
import numpy as np
from utils import *
from InfantryDeathmatch import *
from stable_baselines3 import A2C
import time

policy = {
    "aim_reward_bias": 1,
    "aim_reward_weight": 10,
    "shoot_reward": 100,
    "move_reward": 1,
    "hit_wall_penalty": 0
}

# Initialize the environment
env = InfantryDeathmatch(policy, do_render=False)

model = A2C.load("a2c_robot_game_env", device='cpu')

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

    # time.sleep(0.05)
    counter += 1
    print("Counter: ", counter, "Player1: ", env.player1_hp, "@", env.player1_pos, "Player2: ", env.player2_hp, "@", env.player2_pos, "Detect: ", info["detect"])

env.close()