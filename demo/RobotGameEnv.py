import gym
from gym import spaces
import numpy as np
import math
import pygame
import random as rd

BOARD_X = 3
BOARD_Y = 3
WORLD_TICK = 200
PLAYER_TICK = 200
BULLET_DMG = 0.1

PLAYER1_R = 0.2
PLAYER1_BULLET_SPEED = 30
PLAYER1_HP = 100
PLAYER1_SPEED = 1
PLAYER2_R = 0.2
PLAYER2_BULLET_SPEED = 30
PLAYER2_HP = 100
PLAYER2_SPEED = 1

def normalize(v, len=1):
    assert len != 0
    norm = math.sqrt(v[0] ** 2 + v[1] ** 2) / len
    return (v[0] / norm, v[1] / norm) if norm != 0 else (0, 0)

def get_dist_points(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def get_dist(line, point):
    x0, y0 = point
    x1, y1 = line[0]
    x2, y2 = line[1]
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    distance = numerator / denominator if denominator else math.inf
    return distance

class Bullet:
    def __init__(self, position, direction, speed, owner):
        self.position = position
        self.direction = normalize(direction, speed / WORLD_TICK)
        self.speed = speed
        self.owner = owner
        
    def step(self):
        last_position = self.position
        self.position = (self.position[0] + self.direction[0], self.position[1] + self.direction[1])
        return (last_position, self.position)

class Player:
    def __init__(self, r, bullet_speed, hp, speed, position=(0, 0)):
        self.r = r
        self.bullet_speed = bullet_speed
        self.position = position
        self.hp = hp
        self.speed = speed
        self.tick = 0
        self.action = None

class RobotGameEnv(gym.Env):
    def __init__(self, policy, render=False):
        super(RobotGameEnv, self).__init__()
        self.action_space = spaces.Tuple((
            spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),    # move
            spaces.Discrete(2),     # attack
            spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)     # attack_direction
        ))
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=BOARD_X, shape=(2,), dtype=np.float32),      # position
            spaces.Box(low=0, high=BOARD_X, shape=(2,), dtype=np.float32),      # opponent_position
            spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int64),        # hp
            spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int64),        # radius
            spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int64),        # opponent_hp
            spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int64),        # opponent_radius
            spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int64),        # bullet_speed
        ))
        self.state = None
        self.policy = policy
        self._render = render

        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((600, 600))
            self.clock = pygame.time.Clock()

        self.player1 = Player(PLAYER1_R, PLAYER1_BULLET_SPEED, PLAYER1_HP, PLAYER1_SPEED, (1, 1))
        self.player2 = Player(PLAYER2_R, PLAYER2_BULLET_SPEED, PLAYER2_HP, PLAYER2_SPEED, (2, 2))
        pos = self._random_spawn()
        self.player1.position = pos[0]
        self.player2.position = pos[1]
        self.bullets = []

    def _in_bound(self, pos):
        return pos[0] >= 0 and pos[0] <= BOARD_X and pos[1] >= 0 and pos[1] <= BOARD_Y
    
    def _player_in_bound(self, player, pos):
        corner1 = (pos[0] - player.r, pos[1] - player.r)
        corner2 = (pos[0] + player.r, pos[1] + player.r)
        corner3 = (pos[0] - player.r, pos[1] + player.r)
        corner4 = (pos[0] + player.r, pos[1] - player.r)
        return self._in_bound(corner1) and self._in_bound(corner2) and self._in_bound(corner3) and self._in_bound(corner4)
    
    def _random_spawn(self):
        while True:
            pos1 = (rd.uniform(0, BOARD_X), rd.uniform(0, BOARD_Y))
            pos2 = (rd.uniform(0, BOARD_X), rd.uniform(0, BOARD_Y))
            if self._player_in_bound(self.player1, pos1) and self._player_in_bound(self.player2, pos2) and get_dist_points(pos1, pos2) > self.player1.r + self.player2.r:
                return pos1, pos2

    def step(self, action1, action2):
        reward = [0, 0]
        done = False

        move1 = normalize(action1[0], PLAYER1_SPEED / WORLD_TICK)
        attack1 = action1[1]
        if attack1:
            reward[0] += self.policy["shoot_reward"]
        attack_direction1 = action1[2]

        move2 = normalize(action2[0], PLAYER2_SPEED / WORLD_TICK)
        attack2 = action2[1]
        if attack2:
            reward[1] += self.policy["shoot_reward"]
        attack_direction2 = action2[2]

        for tick in range(WORLD_TICK // PLAYER_TICK):
            next_pos1 = (self.player1.position[0] + move1[0], self.player1.position[1] + move1[1])
            if self._player_in_bound(self.player1, next_pos1):
                self.player1.position = next_pos1
            else:
                reward[0] -= 1

            next_pos2 = (self.player2.position[0] + move2[0], self.player2.position[1] + move2[1])
            if self._player_in_bound(self.player2, next_pos2):
                self.player2.position = next_pos2
            else:
                reward[1] -= 1

            if attack1:
                self.bullets.append(Bullet(self.player1.position, attack_direction1, self.player1.bullet_speed, 1))
            if attack2:
                self.bullets.append(Bullet(self.player2.position, attack_direction2, self.player2.bullet_speed, 2))
            
            for bullet in self.bullets:
                traj = bullet.step()
                opponent = self.player1 if bullet.owner == 2 else self.player2
                if get_dist(traj, opponent.position) < opponent.r:
                    if opponent == self.player1:
                        reward[1] += self.policy["hit_reward"]
                    else:
                        reward[0] += self.policy["hit_reward"]
                    opponent.hp -= BULLET_DMG
                    if opponent.hp <= 0:
                        done = True
                        self.winner = self.player2 if bullet.owner == 2 else self.player1
                        if self.winner == self.player1:
                            reward[0] += self.policy["win_reward"]
                            reward[1] -= self.policy["death_penalty"]
                        else:
                            reward[1] += self.policy["win_reward"]
                            reward[0] -= self.policy["death_penalty"]
                        break
                if not self._in_bound(traj[1]):
                    self.bullets.remove(bullet)
        
        state1 = [self.player1.position, self.player2.position, self.player1.hp, self.player1.r, self.player2.hp, self.player2.r, self.player1.bullet_speed]
        state2 = [self.player2.position, self.player1.position, self.player2.hp, self.player2.r, self.player1.hp, self.player1.r, self.player2.bullet_speed]
        self.state = (state1, state2)

        return self.state, reward, done, {}

    def reset(self):
        # self.state = self.observation_space.sample()
        self.state = [([1, 1], [2, 2], PLAYER1_HP, PLAYER1_R, PLAYER2_HP, PLAYER2_R, PLAYER1_BULLET_SPEED), ([2, 2], [1, 1], PLAYER2_HP, PLAYER2_R, PLAYER1_HP, PLAYER1_R, PLAYER2_BULLET_SPEED)]
        # print("default state: ", self.state)
        return self.state

    def render(self, mode='human'):
        if self._render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

            self.screen.fill((255, 255, 255))

            # Convert positions and radii to integers and scale them
            player1_pos = (int(self.player1.position[0] * 200), int(self.player1.position[1] * 200))
            player2_pos = (int(self.player2.position[0] * 200), int(self.player2.position[1] * 200))
            player1_r = int(self.player1.r * 200)
            player2_r = int(self.player2.r * 200)

            pygame.draw.circle(self.screen, (0, 0, 255), player1_pos, player1_r)
            pygame.draw.circle(self.screen, (255, 0, 0), player2_pos, player2_r)

            for bullet in self.bullets:
                bullet_pos = (int(bullet.position[0] * 200), int(bullet.position[1] * 200))
                pygame.draw.circle(self.screen, (0, 0, 0), bullet_pos, int(0.01 * 200))

            pygame.display.flip()
            self.clock.tick(60)

from gym.envs.registration import register
register(
    id='RobotGameEnv-v0',
    entry_point='RobotGameEnv:RobotGameEnv',
)