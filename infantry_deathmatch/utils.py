import numpy as np
import pygame

# Game Constants
ARENA_SIZE = 5
RESOLUTION = 100    # Split arena into 100x100 grid
TICK = 0.01

# Infantry Constants
AUTOAIMING_ACCURACY = 0.3
MAX_HP = 100
SPEED = 1
R = 0.25
MAX_ROTATION_SPEED = 2 * np.pi
ATTACK_RANGE = max(np.pi / 4, MAX_ROTATION_SPEED * TICK)
HIT_DAMAGE = 1

# Generate Barrier
barrier_pixels = [
    (38, 41),
    (39, 41),
    (41, 41),
    (41, 42),
    (41, 43),
    (41, 44),
    (37, 42),
    (36, 43),
    (35, 44),
    (34, 45),
    (33, 46),
    (32, 47),
    (31, 48),
    (30, 49),
    (29, 50),
    (28, 51),
    (27, 52),
    (26, 52),
    (25, 52),
    (24, 52),
    (23, 52),
    (23, 53),
    (23, 54),
    (23, 55),
    (23, 56),
    (24, 56),
    (25, 56),
    (26, 56),
    (27, 55),
    (28, 54),
    (29, 53),
    (30, 52),
    (31, 51),
    (32, 50),
    (33, 49),
    (34, 48),
    (35, 47),
    (36, 46),
    (37, 45),
    (38, 44),
    (39, 44)
]
barrier_grid = np.zeros((RESOLUTION, RESOLUTION), dtype=bool)
for pixel in barrier_pixels:
    barrier_grid[pixel[1], pixel[0]] = True
for i in range(RESOLUTION):
    row = barrier_grid[i]
    edge = []
    for j in range(RESOLUTION):
        if row[j]:
            edge.append(j)
    if len(edge) < 2:
        continue
    for j in range(edge[0], edge[-1] + 1):
        barrier_grid[i, j] = True
for i in range(RESOLUTION):
    for j in range(RESOLUTION - i):
        barrier_grid[RESOLUTION - 1 - i, RESOLUTION - 1 - j] = barrier_grid[i, j]

def demo_show_pixels(pixels):
    pygame.init()
    screen = pygame.display.set_mode((ARENA_SIZE * RESOLUTION, ARENA_SIZE * RESOLUTION))
    pygame.display.set_caption("Infantry Deathmatch")
    clock = pygame.time.Clock()

    for pixel in pixels:
        pygame.draw.rect(screen, (0, 0, 0), (pixel[0] * ARENA_SIZE, pixel[1] * ARENA_SIZE, ARENA_SIZE, ARENA_SIZE))

    pygame.display.flip()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))  # Clear the screen with a white background

        for pixel in pixels:
            pygame.draw.rect(screen, (0, 0, 0), (pixel[0] * ARENA_SIZE, pixel[1] * ARENA_SIZE, ARENA_SIZE, ARENA_SIZE))

        pygame.display.flip()
        clock.tick(60)  # Limit the frame rate to 60 FPS

    pygame.quit()

def demo_show_grid(grid):
    pygame.init()
    screen = pygame.display.set_mode((ARENA_SIZE * RESOLUTION, ARENA_SIZE * RESOLUTION))
    pygame.display.set_caption("Infantry Deathmatch")
    clock = pygame.time.Clock()

    for x in range(RESOLUTION):
        for y in range(RESOLUTION):
            if grid[y][x]:
                pygame.draw.rect(screen, (0, 0, 0), (x * ARENA_SIZE, y * ARENA_SIZE, ARENA_SIZE, ARENA_SIZE))

    pygame.display.flip()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))  # Clear the screen with a white background

        for x in range(RESOLUTION):
            for y in range(RESOLUTION):
                if grid[y][x]:
                    pygame.draw.rect(screen, (0, 0, 0), (x * ARENA_SIZE, y * ARENA_SIZE, ARENA_SIZE, ARENA_SIZE))

        pygame.display.flip()
        clock.tick(60)  # Limit the frame rate to 60 FPS

    pygame.quit()

def get_hidden_pixels(pos, barrier_grid=barrier_grid):
    barrier_pixels = set()
    for x in range(RESOLUTION):
        for y in range(RESOLUTION):
            if barrier_grid[y][x]:
                barrier_pixels.add((x, y))

    def bresenham_line(x0, y0, x1, y1):
        points = set()
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True:
            points.add((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        return points

    hidden_pixels = set()
    for x in range(RESOLUTION):
        for y in range(RESOLUTION):
            if (x, y) in barrier_pixels:
                hidden_pixels.add((x, y))
            if (x, y) == pos:
                continue
            line = bresenham_line(pos[0], pos[1], x, y)
            for line_pixel in line:
                if line_pixel in barrier_pixels:
                    hidden_pixels.add((x, y))
                    break

    return hidden_pixels

def split_hidden_areas_pixels(hidden_pixels):
    def flood_fill(grid, x, y, visited):
        stack = [(x, y)]
        block = []
        while stack:
            cx, cy = stack.pop()
            if 0 <= cx < grid.shape[1] and 0 <= cy < grid.shape[0] and not visited[cy, cx] and grid[cy, cx]:
                visited[cy, cx] = True
                block.append((cx, cy))
                stack.append((cx + 1, cy))
                stack.append((cx - 1, cy))
                stack.append((cx, cy + 1))
                stack.append((cx, cy - 1))
        return block
    
    def split_continuous_blocks(grid):
        visited = np.zeros_like(grid, dtype=bool)
        blocks = []
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if grid[y, x] and not visited[y, x]:
                    block = flood_fill(grid, x, y, visited)
                    blocks.append(block)
        return blocks

    hidden_grid = np.zeros((RESOLUTION, RESOLUTION), dtype=bool)
    for pixel in hidden_pixels:
        hidden_grid[pixel[1], pixel[0]] = True
    
    hidden_areas = split_continuous_blocks(hidden_grid)
    return hidden_areas

def identify_hidden_area(hidden_areas, pos):
    for hidden_area in hidden_areas:
        if pos in hidden_area:
            return hidden_area
    return None

def rotate_obs(obs):
    obs = list(obs)
    obs[0] = (ARENA_SIZE - obs[0][0], ARENA_SIZE - obs[0][1])
    obs[2] = (obs[2] + np.pi) % (2 * np.pi)
    obs[3] = (obs[3] + np.pi) % (2 * np.pi)
    obs[8] = (ARENA_SIZE - obs[8][0], ARENA_SIZE - obs[8][1])
    grid = obs[9]
    for i in range(RESOLUTION):
        for j in range(RESOLUTION):
            obs[9][RESOLUTION - 1 - i, RESOLUTION - 1 - j] = grid[i, j]
    return obs

def rotate_action(action):
    action[1] = (action[1] + np.pi) % (2 * np.pi)
    action[2] = (action[2] + np.pi) % (2 * np.pi)
    return action

def pixels_to_grid(pixels):
    grid = np.zeros((RESOLUTION, RESOLUTION), dtype=np.uint8)
    try:
        for pixel in pixels:
            grid[pixel[0], pixel[1]] = 1
    except TypeError:
        pass
    return grid

def grid_to_pixels(grid):
    pixels = []
    for x in range(RESOLUTION):
        for y in range(RESOLUTION):
            if grid[x, y]:
                pixels.append((x, y))
    return pixels

def pos_to_pixel(pos):
    return (int(pos[0] / ARENA_SIZE * RESOLUTION), int(pos[1] / ARENA_SIZE * RESOLUTION))

def pixel_to_pos(pixel):
    return (pixel[0] / RESOLUTION * ARENA_SIZE, pixel[1] / RESOLUTION * ARENA_SIZE)

if __name__ == "__main__":
    # demo_show_grid(barrier_grid)
    # demo_show_pixels(get_hidden_pixels((50, 50), barrier_grid))
    hidden_areas = split_hidden_areas_pixels(get_hidden_pixels((50, 50), barrier_grid))
    print(len(hidden_areas))
    demo_show_pixels(hidden_areas[1])