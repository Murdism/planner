#!/usr/bin/env python3
"""
Improved D* Lite grid planner (cleaned + fixes + visualization)
"""

import heapq
import copy
import math
from typing import List, Tuple, Set, Optional, Dict
from dataclasses import dataclass
import numpy as np
import time

ROS2_AVAILABLE = False  # Force standalone mode for testing

@dataclass(frozen=True)
class GridNode:
    x: int
    y: int

    def __add__(self, other):
        return GridNode(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return GridNode(self.x - other.x, self.y - other.y)


@dataclass
class Motion:
    dx: int
    dy: int
    cost: float

    def to_node(self) -> GridNode:
        return GridNode(self.dx, self.dy)


def motions_grid8() -> List[Motion]:
    """8-connected grid motions (cardinal + diagonal)."""
    card = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    diag = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    return [Motion(dx, dy, 1.0) for dx, dy in card] + \
           [Motion(dx, dy, math.sqrt(2.0)) for dx, dy in diag]


def octile_heuristic(a: GridNode, b: GridNode) -> float:
    dx = abs(a.x - b.x)
    dy = abs(a.y - b.y)
    dmax, dmin = (dx, dy) if dx >= dy else (dy, dx)
    return dmax + (math.sqrt(2.0) - 1.0) * dmin


def edge_cost(cell_cost: float, geo_step: float, alpha: float = 1.0) -> float:
    return geo_step * (1.0 + alpha * cell_cost)


class PriorityQueue:
    """Priority queue with lazy deletion and stable tiebreaker."""

    def __init__(self):
        self._queue = []  # entries are [priority_tuple, count, task]
        self._entry_finder = {}  # task -> entry
        self._counter = 0
        self.REMOVED = object()  # unique token

    def add_task(self, task: GridNode, priority: Tuple[float, float]):
        if task in self._entry_finder:
            self.remove_task(task)
        count = self._counter
        self._counter += 1
        entry = [priority, count, task]
        self._entry_finder[task] = entry
        heapq.heappush(self._queue, entry)

    def remove_task(self, task: GridNode):
        entry = self._entry_finder.pop(task, None)
        if entry is not None:
            entry[2] = self.REMOVED

    def pop_task(self) -> Tuple[GridNode, Tuple[float, float]]:
        while self._queue:
            priority, count, task = heapq.heappop(self._queue)
            if task is not self.REMOVED:
                self._entry_finder.pop(task, None)
                return task, priority
        raise KeyError('pop from an empty priority queue')

    def top_key(self) -> Optional[Tuple[float, float]]:
        while self._queue and self._queue[0][2] is self.REMOVED:
            heapq.heappop(self._queue)
        return self._queue[0][0] if self._queue else None

    def empty(self) -> bool:
        return len(self._entry_finder) == 0


class DStarLite:
    """D* Lite incremental planner."""

    def __init__(self, width: int, height: int, obstacles: Set[Tuple[int, int]] = None,
                 alpha: float = 1.0):
        self.width = width
        self.height = height
        self.obstacles = set(obstacles or set())
        self.dynamic_obstacles: Set[Tuple[int, int]] = set()
        self.cell_costs: Dict[GridNode, float] = {}
        self.alpha = alpha

        self.start = GridNode(0, 0)
        self.goal = GridNode(0, 0)
        self.last_start = GridNode(0, 0)
        self.km = 0.0

        self.g_values: Dict[GridNode, float] = {}
        self.rhs_values: Dict[GridNode, float] = {}
        self.U = PriorityQueue()

        self.MOTIONS = motions_grid8()
        self.initialized = False

    # ----------------------------- basic accessors -----------------------------
    def get_g(self, node: GridNode) -> float:
        return self.g_values.get(node, math.inf)

    def set_g(self, node: GridNode, value: float):
        if value == math.inf:
            self.g_values.pop(node, None)
        else:
            self.g_values[node] = value

    def get_rhs(self, node: GridNode) -> float:
        return self.rhs_values.get(node, math.inf)

    def set_rhs(self, node: GridNode, value: float):
        if value == math.inf:
            self.rhs_values.pop(node, None)
        else:
            self.rhs_values[node] = value

    def is_valid(self, node: GridNode) -> bool:
        return 0 <= node.x < self.width and 0 <= node.y < self.height

    def is_obstacle(self, node: GridNode) -> bool:
        return (node.x, node.y) in self.obstacles or (node.x, node.y) in self.dynamic_obstacles

    def get_neighbors(self, node: GridNode) -> List[GridNode]:
        neighbors = []
        for motion in self.MOTIONS:
            nb = node + motion.to_node()
            if self.is_valid(nb):
                neighbors.append(nb)
        return neighbors
    def get_successors(self, node: GridNode) -> List[GridNode]:
        return self.get_neighbors(node)

    def cost(self, from_node: GridNode, to_node: GridNode) -> float:
        if self.is_obstacle(to_node):
            return math.inf
        diff = to_node - from_node
        for motion in self.MOTIONS:
            if motion.to_node() == diff:
                cell_cost = self.cell_costs.get(to_node, 0.0)
                return edge_cost(cell_cost, motion.cost, alpha=self.alpha)
        return math.inf

    # ----------------------------- heuristics & keys -----------------------------
    def heuristic_between(self, a: GridNode, b: GridNode) -> float:
        return octile_heuristic(a, b)

    def heuristic(self, node: GridNode) -> float:
        return self.heuristic_between(node, self.start)

    def calculate_key(self, node: GridNode) -> Tuple[float, float]:
        g_val = self.get_g(node)
        rhs_val = self.get_rhs(node)
        min_val = min(g_val, rhs_val)
        k1 = min_val + self.heuristic(node) + self.km
        k2 = min_val
        return (k1, k2)

    @staticmethod
    def compare_keys(key1: Tuple[float, float], key2: Tuple[float, float]) -> bool:
        return key1[0] < key2[0] or (key1[0] == key2[0] and key1[1] < key2[1])

    # ----------------------------- core D* Lite ops -----------------------------
    def update_vertex(self, node: GridNode):
        if node != self.goal:
            rhs_candidates = [self.cost(node, succ) + self.get_g(succ)
                              for succ in self.get_neighbors(node)
                              if self.cost(node, succ) < math.inf]
            self.set_rhs(node, min(rhs_candidates) if rhs_candidates else math.inf)
        self.U.remove_task(node)
        if self.get_g(node) != self.get_rhs(node):
            self.U.add_task(node, self.calculate_key(node))

    def compute_shortest_path(self):
        iterations = 0
        while (not self.U.empty() and
               (self.compare_keys(self.U.top_key(), self.calculate_key(self.start)) or
                self.get_rhs(self.start) != self.get_g(self.start))):

            iterations += 1
            if iterations > 10000:
                print("WARNING: compute_shortest_path exceeded iteration limit")
                break

            node, k_old = self.U.pop_task()
            k_new = self.calculate_key(node)

            if self.compare_keys(k_old, k_new):
                self.U.add_task(node, k_new)
            elif self.get_g(node) > self.get_rhs(node):
                self.set_g(node, self.get_rhs(node))
                for pred in self.get_neighbors(node):
                    self.update_vertex(pred)
            else:
                self.set_g(node, math.inf)
                for pred in self.get_neighbors(node) + [node]:
                    self.update_vertex(pred)

    # ----------------------------- initialization / external API -----------------------------
    def initialize(self, start: GridNode, goal: GridNode):
        self.start = start
        self.goal = goal
        self.last_start = start
        if not self.initialized:
            self.initialized = True
            self.U = PriorityQueue()
            self.km = 0.0
            self.g_values.clear()
            self.rhs_values.clear()
            self.set_rhs(self.goal, 0.0)
            self.U.add_task(self.goal, self.calculate_key(self.goal))

    def set_start(self, new_start: GridNode):
        self.km += self.heuristic_between(self.last_start, new_start)
        self.last_start = new_start
        self.start = new_start
        self.update_vertex(self.start)
        self.compute_shortest_path()

    def set_goal(self, new_goal: GridNode):
        self.goal = new_goal
        self.set_rhs(self.goal, 0.0)
        self.U.add_task(self.goal, self.calculate_key(self.goal))
        self.compute_shortest_path()

    def plan_path(self, start: GridNode, goal: GridNode) -> List[GridNode]:
        if not self.initialized:
            self.initialize(start, goal)
        else:
            if start != self.start:
                self.set_start(start)
            if goal != self.goal:
                self.set_goal(goal)
        self.compute_shortest_path()
        return self.extract_path() if self.get_g(self.start) < math.inf else []

    def extract_path(self) -> List[GridNode]:
        """Extract the best path from start to goal using current g-values."""
        if self.get_g(self.start) == math.inf:
            return []

        path: List[GridNode] = []
        current = self.start
        visited = set()
        max_iterations = self.width * self.height

        iteration_count = 0
        while current != self.goal and iteration_count < max_iterations:
            path.append(current)
            visited.add(current)
            iteration_count += 1

            # Choose the neighbor with the minimum cost to goal
            best_succ = None
            best_cost = math.inf
            for succ in self.get_successors(current):
                if self.is_obstacle(succ) or succ in visited:
                    continue
                val = self.cost(current, succ) + self.get_g(succ)
                if val < best_cost:
                    best_cost = val
                    best_succ = succ

            if best_succ is None:
                # Don't break or replan here; just stop path
                break

            current = best_succ

        if current == self.goal:
            path.append(self.goal)

        return path

    def update_dynamic_obstacles(self, added: Set[Tuple[int, int]], removed: Set[Tuple[int, int]]):
        changed_nodes = set()
        for o in added:
            if o not in self.dynamic_obstacles:
                self.dynamic_obstacles.add(o)
                changed_nodes.add(GridNode(*o))
        for o in removed:
            if o in self.dynamic_obstacles:
                self.dynamic_obstacles.remove(o)
                changed_nodes.add(GridNode(*o))

        if changed_nodes:
            # Update km for distance traveled
            self.km += self.heuristic_between(self.last_start, self.start)
            self.last_start = self.start

            # Mark all changed nodes as needing update
            for node in changed_nodes:
                # Force g/rhs to infinity so update_vertex recomputes them
                self.set_g(node, math.inf)
                self.set_rhs(node, math.inf)
                self.update_vertex(node)

            # Recompute the shortest path
            self.compute_shortest_path()



# ------------------------- visualization -------------------------
def visualize_grid(planner: DStarLite, path: List[GridNode] = None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    grid = np.zeros((planner.height, planner.width), dtype=np.uint8)
    for (x, y) in planner.obstacles:
        grid[y, x] = 200
    for (x, y) in planner.dynamic_obstacles:
        grid[y, x] = 120
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid, origin='lower', cmap='gray_r')
    if path:
        xs = [n.x for n in path]
        ys = [n.y for n in path]
        ax.plot(xs, ys, 'o-', linewidth=2)
    ax.scatter([planner.start.x], [planner.start.y], c='green', s=100, marker='*', label='start')
    ax.scatter([planner.goal.x], [planner.goal.y], c='red', s=100, marker='X', label='goal')
    ax.set_xlim(-0.5, planner.width - 0.5)
    ax.set_ylim(-0.5, planner.height - 0.5)
    ax.legend()
    plt.show()


# ------------------------- standalone demo -------------------------
def demo_standalone():
    print("Running D* Lite standalone demo...")
    width, height = 50, 50
    obstacles = {(20, y) for y in range(5, 25)}
    planner = DStarLite(width, height, obstacles)
    start = GridNode(2, 10)
    goal = GridNode(38, 15)

    path = planner.plan_path(start, goal)
    if not path:
        print("No initial path found")
        return
    print(f"Initial path length: {len(path)}")
    visualize_grid(planner, path)

    for i in range(5):
        added = {(10 + i, 6 + i), (int(11 * (i / 3)), 2), (int(12 + 2 * i), 10 - i)}
        print(f"Adding dynamic obstacles: {added}")
        planner.update_dynamic_obstacles(added, set())
        path = planner.extract_path()
        if path:
            print(f"Replanned path length: {len(path)}")
            visualize_grid(planner, path)
        else:
            print("No path found after dynamic obstacles")


def main():
    if not ROS2_AVAILABLE:
        demo_standalone()


if __name__ == '__main__':
    main()
