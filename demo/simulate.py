import gym
import numpy as np
from RobotGameEnv import RobotGameEnv, PLAYER_TICK
from stable_baselines3 import PPO, TD3
import time

policy = {
    "shoot_reward": 10,
    "hit_reward": 100,
    "win_reward": 1000,
    "death_penalty": 0
}

# Initialize the environment
env = RobotGameEnv(policy, render=True)

model = PPO.load("ppo_robot_game_env", device='cpu')

done = False
env.reset()
counter = 0
while not done:
    # Infer action1 using the pre-trained model
    raw_obs = env.state[0]  # Get the observation for the second agent
    obs = np.concatenate([np.array(o).flatten() for o in raw_obs])
    action1, _ = model.predict(obs)
    move1 = action1[0:2]
    attack1 = 1 if action1[2] >= 0.5 else 0
    attack_direction1 = action1[3:5]
    formatted_action1 = (move1, attack1, attack_direction1)

    # Infer action2 using the pre-trained model
    raw_obs = env.state[1]  # Get the observation for the second agent
    obs = np.concatenate([np.array(o).flatten() for o in raw_obs])
    action2, _ = model.predict(obs)
    move2 = action2[0:2]
    attack2 = 1 if action2[2] >= 0.5 else 0
    attack_direction2 = action2[3:5]
    formatted_action2 = (move2, attack2, attack_direction2)

    obs, reward, done, info = env.step(formatted_action1, formatted_action2)

    # Render the game
    env.render()

    time.sleep(0.05)
    counter += 1
    print("Counter: ", counter, "Player1 HP: ", env.player1.hp, "Player2 HP: ", env.player2.hp)

env.close()