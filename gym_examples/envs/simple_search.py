import gymnasium as gym
from gymnasium import spaces

from typing import Optional

import numpy as np

class SimpleSearchEnv(gym.Env):
    def __init__(self, render_mode: Optional[str] = None, search_tree_depth=4, patience_penalty=1.0, cognitive_slowness=3):
        self.render_mode = render_mode
        self.search_tree_depth = search_tree_depth
        self.max_episode_steps = 1000
        self.patience_penalty = patience_penalty
        self.cognitive_slowness = cognitive_slowness

        self.binary_search_tree = self._generate_binary_search_tree(search_tree_depth)
        self.current_node = self.binary_search_tree
        self.time_until_knowing_node = self.cognitive_slowness
        self.time_step = 0

        self.state = self._get_fuzzy_observations()

        # actions
        # 0 go to home
        # 1 go to parent
        # 2 go left
        # 3 go right
        # 4 stay (examine current node)
        # 5 choose current node (end search)
        self.action_space = spaces.Discrete(6)

        # observations
        # 0 value in current node
        # 1 fuzzy value in left node
        # 2 fuzzy value in right node
        self.observation_space = spaces.Box(low=0, high=100, shape=(3,), dtype=np.float32)

    def step(self, action: int):
        self.time_step += 1
        # penalize each time step (to encourage quick search)
        reward = -self.patience_penalty
        terminated = False

        if (action == 5) or (self.time_step >= self.max_episode_steps): # choose current node (end search)
            reward += self.current_node.value
            terminated = True
        elif action == 0: # go to home
            self.current_node = self.binary_search_tree
            self.time_until_knowing_node = self.cognitive_slowness
        elif action == 1: # go to parent
            if self.current_node.parent is not None:
                self.current_node = self.current_node.parent
                self.time_until_knowing_node = self.cognitive_slowness
        elif action == 2: # go left
            if self.current_node.left is not None:
                self.current_node = self.current_node.left
                self.time_until_knowing_node = self.cognitive_slowness
        elif action == 3: # go right
            if self.current_node.right is not None:
                self.current_node = self.current_node.right
                self.time_until_knowing_node = self.cognitive_slowness
        elif action == 4: # stay (examine current node)
            if self.time_until_knowing_node > 0:
                self.time_until_knowing_node -= 1

        self.state = self._get_fuzzy_observations()
    
        return self.state, reward, terminated, False, {}
        

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        self.time_until_knowing_node = self.cognitive_slowness
        self.time_step = 0
        self.binary_search_tree = self._generate_binary_search_tree(self.search_tree_depth)
        self.current_node = self.binary_search_tree
        self.state = self._get_fuzzy_observations()

        return self.state, {}

    def _get_fuzzy_observations(self):
        current_value = self.current_node.value + np.random.normal(0, self.time_until_knowing_node)
        left_value = -1
        right_value = -1
        if self.current_node.left is not None:
            left_value = self.current_node.left.value + np.random.normal(0, self.cognitive_slowness + 1)
        if self.current_node.right is not None:
            right_value = self.current_node.right.value + np.random.normal(0, self.cognitive_slowness + 1)

        return np.array([current_value, left_value, right_value], dtype=np.float32)

    def render(self):
        pass

    def log(self):
        print(f"Time step: {self.time_step}, Observarion (current, left, right): {self.state}, Current node real value: {self.current_node.value}")

    def get_current_node_value(self):
        if self.current_node is not None:
            return self.current_node.value

    def _generate_binary_search_tree(self, depth):
        binary_tree = BinaryTree(50)
        for _ in range(depth):
            binary_tree.insert()
        #print("binary tree:")
        #print(binary_tree.to_string())
        return binary_tree

    def close(self):
        pass

class BinaryTree():
    def __init__(self, value, parent=None):
        self.value = value
        self.left = None
        self.right = None
        self.parent = parent

    def insert(self):
        new_value = np.random.normal(self.value, 10)
        if self.left is None:
            self.left = BinaryTree(new_value, self)
        else:
            self.left.insert()

        new_value = np.random.normal(self.value, 10)
        if self.right is None:
            self.right = BinaryTree(new_value, self)
        else:
            self.right.insert()

    def to_string(self):
        left = self.left.to_string() if self.left is not None else ""
        right = self.right.to_string() if self.right is not None else ""
        return f"({self.value} l:{left} r:{right})"
