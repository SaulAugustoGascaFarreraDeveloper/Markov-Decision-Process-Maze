from typing import Tuple, Dict, Iterable, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import pygame
from pygame import gfxdraw


class Maze(gym.Env):
    """
    Grid-world Maze environment for Reinforcement Learning.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        exploring_starts: bool = False,
        shaped_rewards: bool = False,
        size: int = 5,
    ) -> None:
        super().__init__()

        self.size = size
        self.exploring_starts = exploring_starts
        self.shaped_rewards = shaped_rewards

        self.state = (0, 0)
        self.goal = (size - 1, size - 1)

        self.maze = self._create_maze(size)
        self.distances = self._compute_distances(self.goal, self.maze)

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([size, size])

        self.screen = None

    # ==============================
    # Gym API
    # ==============================

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.exploring_starts:
            while self.state == self.goal:
                self.state = tuple(self.observation_space.sample())
        else:
            self.state = (0, 0)

        return self.state, {}

    def step(self, action: int):
        reward = self.compute_reward(self.state, action)
        self.state = self._get_next_state(self.state, action)
        terminated = self.state == self.goal
        truncated = False
        info = {}

        return self.state, reward, terminated, truncated, info

    # ==============================
    # Rendering
    # ==============================

    


    def render(self, mode="human") -> Optional[np.ndarray]:
        assert mode in ["human", "rgb_array"]

        screen_size = 600
        scale = screen_size / self.size

        wall_thickness = 3

        if self.screen is None:
            pygame.init()
            self.screen = pygame.Surface((screen_size, screen_size))

        surf = pygame.Surface((screen_size, screen_size))
        surf.fill((22, 36, 71))

      


        # Draw walls
        for row in range(self.size):
            for col in range(self.size):
                state = (row, col)
                for next_state in [
                    (row + 1, col),
                    (row - 1, col),
                    (row, col + 1),
                    (row, col - 1),
                ]:
                    if next_state not in self.maze[state]:
                        row_diff, col_diff = np.subtract(next_state, state)

                        #left = (col + (col_diff > 0)) * scale
                        #right = ((col + 1) - (col_diff < 0)) * scale
                        #top = (self.size - (row + (row_diff > 0))) * scale
                        #bottom = (self.size - ((row + 1) - (row_diff < 0))) * scale

                        left = (col + (col_diff > 0)) * scale - wall_thickness
                        right = ((col + 1) - (col_diff < 0)) * scale + wall_thickness
                        top = (self.size - (row + (row_diff > 0))) * scale - wall_thickness
                        bottom = (self.size - ((row + 1) - (row_diff < 0))) * scale + wall_thickness


                        gfxdraw.filled_polygon(
                            surf,
                            [(left, bottom), (left, top), (right, top), (right, bottom)],
                            (255, 255, 255),
                        )

        # Goal
        left = scale * (self.size - 1) + 10
        right = scale * self.size - 10
        top = scale - 10
        bottom = 10

        gfxdraw.filled_polygon(
            surf,
            [(left, bottom), (left, top), (right, top), (right, bottom)],
            (40, 199, 172),
        )

        # Agent
        #agent_row = int(screen_size - scale * (self.state[0] + 0.5))
        #agent_col = int(scale * (self.state[1] + 0.5))

        #agent_row = int(scale * (self.size - 1 - self.state[0] + 0.5))
        #agent_col = int(scale * (self.state[1] + 0.5))

        #agent_row = int(scale * (self.state[0] + 0.5))
        #agent_col = int(scale * (self.state[1] + 0.5))

        agent_row = int(scale * (self.size - 1 - self.state[0] + 0.5))
        agent_col = int(scale * (self.state[1] + 0.5))



        gfxdraw.filled_circle(
            surf,
            agent_col,
            agent_row,
            int(scale * 0.3),
            (228, 63, 90),
        )

        #self.screen.blit(surf, (0, 0))
        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0, 0))


        if mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

        return None

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    # ==============================
    # Logic
    # ==============================

    def compute_reward(self, state: Tuple[int, int], action: int) -> float:
        next_state = self._get_next_state(state, action)
        if self.shaped_rewards:
            return -self.distances[next_state] / self.distances.max()
        return -1.0 if state != self.goal else 0.0

    def _get_next_state(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        moves = {
            0: (-1, 0),  # UP
            1: (0, 1),   # RIGHT
            2: (1, 0),   # DOWN
            3: (0, -1),  # LEFT
        }

        dr, dc = moves[action]
        next_state = (state[0] + dr, state[1] + dc)

        if next_state in self.maze[state]:
            return next_state
        return state

    # ==============================
    # Maze construction
    # ==============================

    @staticmethod
    def _create_maze(size: int) -> Dict[Tuple[int, int], Iterable[Tuple[int, int]]]:
        maze = {}
    
        for r in range(size):
            for c in range(size):
                neighbors = []
                for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size:
                        neighbors.append((nr, nc))
                maze[(r, c)] = neighbors
    
        walls = [
            [(1, 0), (1, 1)], [(2, 0), (2, 1)], [(3, 0), (3, 1)],
            [(1, 1), (1, 2)], [(2, 1), (2, 2)], [(3, 1), (3, 2)],
            [(3, 1), (4, 1)], [(0, 2), (1, 2)], [(1, 2), (1, 3)],
            [(2, 2), (3, 2)], [(2, 3), (3, 3)], [(2, 4), (3, 4)],
            [(4, 2), (4, 3)], [(1, 3), (1, 4)], [(2, 3), (2, 4)],
        ]
    
        for a, b in walls:
            if b in maze[a]:
                maze[a].remove(b)
            if a in maze[b]:
                maze[b].remove(a)
    
        return maze


    @staticmethod
    def _compute_distances(
        goal: Tuple[int, int],
        maze: Dict[Tuple[int, int], Iterable[Tuple[int, int]]],
    ) -> np.ndarray:
        size = int(np.sqrt(len(maze)))
        distances = np.full((size, size), np.inf)
        distances[goal] = 0.0
        visited = set()

        while visited != set(maze):
            flat = distances.argsort(axis=None)
            for v in flat:
                cell = (v // size, v % size)
                if cell not in visited:
                    current = cell
                    break

            visited.add(current)

            for n in maze[current]:
                distances[n] = min(distances[n], distances[current] + 1)

        return distances
