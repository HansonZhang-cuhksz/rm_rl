from InfantryDeathmatch import is_valid_position
from utils import *
import random as rd
import numpy as np
from scipy.optimize import minimize

pos = None
opponent_area_pixels = None
aim_range = (0, 2 * np.pi)

def possible_area_range(direction):
    global pos, opponent_area_pixels, aim_range

    if direction < 0 or direction > 2 * np.pi:
        return np.inf
    next_pos = (pos[0] + np.cos(direction) * SPEED * TICK, pos[1] + np.sin(direction) * SPEED * TICK)
    if not is_valid_position(next_pos):
        return np.inf
    hidden_pixels = get_hidden_pixels(pos_to_pixel(next_pos), barrier_grid)
    hidden_areas = split_hidden_areas_pixels(hidden_pixels)
    next_opponent_area = None
    for hidden_area in hidden_areas:
        for pixel in hidden_area:
            if pixel in opponent_area_pixels:
                next_opponent_area = hidden_area
                break
    assert next_opponent_area is not None
    
    angles = []
    for pixel in next_opponent_area:
        angle = np.arctan2(pixel[1] - pos[1], pixel[0] - pos[0])
        angles.append(angle)
    
    def calculate_angle(x0, y0, x, y):
        return np.arctan2(y - y0, x - x0) % (2 * np.pi)

    def find_minimum_covering_angle(pos, points):
        x0 = pos[0]
        y0 = pos[1]

        angles = [calculate_angle(x0, y0, x, y) for x, y in points]
        angles.sort()

        min_angle = 2 * np.pi
        start_angle = 0
        end_angle = 0

        for i in range(len(angles)):
            current_angle = angles[i]
            next_angle = angles[(i + 1) % len(angles)]
            diff = (next_angle - current_angle) % (2 * np.pi)
            if diff < min_angle:
                min_angle = diff
                start_angle = current_angle
                end_angle = next_angle

        return min_angle, start_angle, end_angle
    
    min_angle, start_angle, end_angle = find_minimum_covering_angle(pos, next_opponent_area)
    aim_range = (start_angle, end_angle)
    return min_angle


def dodge(pos):
    while True:
        direction = rd.uniform(0, 2 * np.pi)
        next_pos = (pos[0] + np.cos(direction) * SPEED * TICK, pos[1] + np.sin(direction) * SPEED * TICK)
        if is_valid_position(next_pos):
            return direction

def decide(obs):
    global pos, opponent_area_pixels, aim_range

    # print("obs", obs)
    pos = obs[0]
    detect = obs[7]

    if detect:
        move = dodge(pos)
        aim_direction = 0
    else:
        opponent_area_grid = obs[9]
        opponent_area_pixels = grid_to_pixels(opponent_area_grid)
        initial = np.pi
        res = minimize(possible_area_range, initial, method='Nelder-Mead')
        possible_area_range(res.x)
        aim_direction = (aim_range[0] + aim_range[1]) / 2
        move = res.x

    return [1, aim_direction, move]