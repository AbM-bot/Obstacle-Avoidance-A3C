import numpy as np

class GridWorld:
    def __init__(self, size=6):
        self.size = size
        self.reset()

    def reset(self):
        self.agent = [0, 0]
        self.goal = [5, 5]
        self.obstacles = [[2, 2], [3, 3], [1, 4]]
        return self.get_state()

    def get_state(self):
        x, y = self.agent

        up = x
        down = self.size - 1 - x
        left = y
        right = self.size - 1 - y

        return [x, y, up, down, left, right]

    def step(self, action):
        if action == 0:
            self.agent[0] -= 1
        elif action == 1:
            self.agent[0] += 1
        elif action == 2:
            self.agent[1] -= 1
        elif action == 3:
            self.agent[1] += 1

        self.agent[0] = max(0, min(self.agent[0], self.size - 1))
        self.agent[1] = max(0, min(self.agent[1], self.size - 1))

        reward = -0.1
        done = False

        goal_distance = abs(self.agent[0] - self.goal[0]) + abs(self.agent[1] - self.goal[1])
        reward += -0.2 * goal_distance

        if self.agent in self.obstacles:
            reward = -20
            done = True

        if self.agent == self.goal:
            reward = 100
            done = True

        return self.get_state(), reward, done