import gym
from gym import spaces
import numpy as np
import random as rd

from utils import *

def is_valid_position(pos):
    if pos[0] - R < 0 or pos[0] + R > ARENA_SIZE or pos[1] - R < 0 or pos[1] + R > ARENA_SIZE:
        return False
    valid = True
    for x in range(RESOLUTION):
        for y in range(RESOLUTION):
            if barrier_grid[x, y]:
                if (x - pos[0] * RESOLUTION) ** 2 + (y - pos[1] * RESOLUTION) ** 2 <= R ** 2:
                    valid = False
    return valid

class InfantryDeathmatch(gym.Env):
    def __init__(self, policy, do_render=False):
        super().__init__()
        self.action_space = spaces.Tuple((
            # spaces.Discrete(2), # move
            spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),   # move
            spaces.Box(low=0, high=2*np.pi, shape=(1,), dtype=np.float32), # move direction
            spaces.Box(low=0, high=2*np.pi, shape=(1,), dtype=np.float32) # aim direction
        ))
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=ARENA_SIZE, shape=(2,), dtype=np.float32), # self position
            # spaces.Discrete(2), # self move
            spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),  # self move
            spaces.Box(low=0, high=2*np.pi, shape=(1,), dtype=np.float32), # self move direction
            spaces.Box(low=0, high=2*np.pi, shape=(1,), dtype=np.float32), # self aim direction
            # spaces.Discrete(2), # hit wall
            spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),  # hit wall
            spaces.Box(low=0, high=MAX_HP, shape=(1,), dtype=np.float32), # self hp
            # spaces.Discrete(2), # get hit
            # spaces.Discrete(2), # opponent detected
            spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),  # get hit
            spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),  # opponent detected
            spaces.Box(low=0, high=ARENA_SIZE, shape=(2,), dtype=np.float32), # opponent position
            spaces.Box(low=0, high=1, shape=(RESOLUTION, RESOLUTION), dtype=np.uint8) # opponent possible positions
        ))

        self.policy = policy

        self.reset()

        self.do_render = do_render
        if do_render:
            pygame.init()
            self.screen = pygame.display.set_mode((RESOLUTION*10, RESOLUTION*10))
            pygame.display.set_caption("Infantry Deathmatch")
            self.clock = pygame.time.Clock()

    def reset(self):
        self.player1_hp = MAX_HP
        self.player2_hp = MAX_HP
        self.player1_pos = (0.5, 0.5)
        self.player2_pos = (4.5, 4.5)
        self.player1_aim = np.pi / 4
        self.player2_aim = 5 * np.pi / 4
        self.player2_possible_positions_for_player1 = get_hidden_pixels(pos_to_pixel(self.player1_pos))
        self.player1_possible_positions_for_player2 = get_hidden_pixels(pos_to_pixel(self.player2_pos))

        predict_grid1 = pixels_to_grid(self.player2_possible_positions_for_player1)
        predict_grid2 = pixels_to_grid(self.player1_possible_positions_for_player2)
        
        # player1_obs = (self.player1_pos, 0, None, self.player1_aim, False, self.player1_hp, False, False, (None, None), predict_grid1)
        # player2_obs = (self.player2_pos, 0, None, self.player2_aim, False, self.player2_hp, False, False, (None, None), predict_grid2)
        player1_obs = (self.player1_pos, 0, 0, self.player1_aim, False, self.player1_hp, False, False, (0, 0), predict_grid1)
        player2_obs = (self.player2_pos, 0, 0, self.player2_aim, False, self.player2_hp, False, False, (0, 0), predict_grid2)

        return player1_obs, player2_obs

    def step(self, action1, action2):
        reward1 = 0
        reward2 = 0

        player1_get_hit = False
        player2_get_hit = False
        player1_hit_wall = False
        player2_hit_wall = False

        # Check detection
        player1_hidden_area = get_hidden_pixels(pos_to_pixel(self.player1_pos))
        detect = pos_to_pixel(self.player2_pos) in player1_hidden_area

        # Orientation
        if detect:
            player2_direction_to_player1 = np.arctan2(self.player1_pos[1] - self.player2_pos[1], self.player1_pos[0] - self.player2_pos[0])
            # Reward for accurate aim
            reward1 += (np.cos(player2_direction_to_player1 - self.player1_aim) + self.policy["aim_reward_bias"]) * self.policy["aim_reward_weight"]
            if player2_direction_to_player1 > self.player1_aim - ATTACK_RANGE / 2 and player2_direction_to_player1 < self.player1_aim + ATTACK_RANGE / 2:
                # Reward for hit
                reward1 += self.policy["shoot_reward"]
                self.player1_aim = player2_direction_to_player1
                if rd.random() < AUTOAIMING_ACCURACY:
                    self.player2_hp -= HIT_DAMAGE
                    player2_get_hit = True
            else:
                if player2_direction_to_player1 - self.player1_aim < np.pi:
                    rotate_direction = -1
                else:
                    rotate_direction = 1
                self.player1_aim += rotate_direction * MAX_ROTATION_SPEED * TICK

            player1_direction_to_player2 = np.arctan2(self.player2_pos[1] - self.player1_pos[1], self.player2_pos[0] - self.player1_pos[0])
            # Reward for accurate aim
            reward2 += (np.cos(player1_direction_to_player2 - self.player2_aim) + self.policy["aim_reward_bias"]) * self.policy["aim_reward_weight"]
            if player1_direction_to_player2 > self.player2_aim - ATTACK_RANGE / 2 and player1_direction_to_player2 < self.player2_aim + ATTACK_RANGE / 2:
                # Reward for hit
                reward2 += self.policy["shoot_reward"]
                self.player2_aim = player1_direction_to_player2
                if rd.random() < AUTOAIMING_ACCURACY:
                    self.player1_hp -= HIT_DAMAGE
                    player1_get_hit = True
            else:
                if player1_direction_to_player2 - self.player2_aim < np.pi:
                    rotate_direction = -1
                else:
                    rotate_direction = 1
                self.player2_aim += rotate_direction * MAX_ROTATION_SPEED * TICK
        else:
            self.player1_aim = action1[2]
            self.player2_aim = action2[2]

        # Move
        if action1[0] >= 0.5:
            # Reward for moving
            reward1 += self.policy["move_reward"]
            player1_new_pos = (self.player1_pos[0] + SPEED * np.cos(action1[1]), self.player1_pos[1] + SPEED * np.sin(action1[1]))
            if is_valid_position(player1_new_pos):
                self.player1_pos = player1_new_pos
            else:
                # Penalty for hitting wall
                reward1 -= self.policy["hit_wall_penalty"]
                player1_hit_wall = True
        
        if action2[0] >= 0.5:
            # Reward for moving
            reward2 += self.policy["move_reward"]
            player2_new_pos = (self.player2_pos[0] + SPEED * np.cos(action2[1]), self.player2_pos[1] + SPEED * np.sin(action2[1]))
            if is_valid_position(player2_new_pos):
                self.player2_pos = player2_new_pos
            else:
                # Penalty for hitting wall
                reward2 -= self.policy["hit_wall_penalty"]
                player2_hit_wall = True

        # Output
        player1_predict_player2_pos = identify_hidden_area(split_hidden_areas_pixels(get_hidden_pixels(pos_to_pixel(self.player1_pos))), pos_to_pixel(self.player2_pos)) if not detect else np.zeros((RESOLUTION, RESOLUTION), dtype=np.uint8)
        player2_predict_player1_pos = identify_hidden_area(split_hidden_areas_pixels(get_hidden_pixels(pos_to_pixel(self.player2_pos))), pos_to_pixel(self.player1_pos)) if not detect else np.zeros((RESOLUTION, RESOLUTION), dtype=np.uint8)
        predict_grid1 = pixels_to_grid(player1_predict_player2_pos)
        predict_grid2 = pixels_to_grid(player2_predict_player1_pos)
        player1_obs = (self.player1_pos, action1[0], action1[1], self.player1_aim, player1_hit_wall, self.player1_hp, player1_get_hit, detect, self.player2_pos if detect else (0, 0), predict_grid1)
        player2_obs = (self.player2_pos, action2[0], action2[1], self.player2_aim, player2_hit_wall, self.player2_hp, player2_get_hit, detect, self.player1_pos if detect else (0, 0), predict_grid2)
        return (player1_obs, player2_obs), (reward1, reward2), self.player1_hp <= 0 or self.player2_hp <= 0, {}
    
    def render(self):
        if self.do_render:
            self.screen.fill((255, 255, 255))
            for x in range(RESOLUTION):
                for y in range(RESOLUTION):
                    if barrier_grid[x, y]:
                        pygame.draw.rect(self.screen, (255, 255, 255), (x, y, 10, 10))
            pygame.draw.circle(self.screen, (0, 255, 0), (self.player1_pos[0] * 200, self.player1_pos[1] * 200), R*200)
            pygame.draw.circle(self.screen, (255, 0, 0), (self.player2_pos[0] * 200, self.player2_pos[1] * 200), R*200)

            # Draw HP bars
            self.draw_hp_bar((self.player1_pos[0] * 200, self.player1_pos[1] * 200 - 50), self.player1_hp, (0, 255, 0))
            self.draw_hp_bar((self.player2_pos[0] * 200, self.player2_pos[1] * 200 - 50), self.player2_hp, (255, 0, 0))

            pygame.display.flip()
            self.clock.tick(60)
            