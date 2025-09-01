#!/usr/bin/env python3
"""
Fixed D* Lite grid planner with proper dynamic obstacle handling

Key fixes:
1. Proper km update only when start actually moves
2. Better handling of dynamic obstacle updates
3. Improved path extraction with better successor selection
4. More robust priority queue management
5. Better debugging and validation
"""

import heapq
import copy
import math
from typing import List, Tuple, Set, Optional, Dict
from dataclasses import dataclass
import numpy as np
import time

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
    """
    Convert a normalized cell cost (0..1) into an edge cost.
    geo_step: 1.0 for cardinal, sqrt(2) for diagonal.
    alpha: controls how strongly the cell cost affects the edge.
    """
    return geo_step * (1.0 + alpha * cell_cost)


class PriorityQueue:
    """Priority queue with lazy deletion and stable tiebreaker."""

    def __init__(self):
        self._queue = []  # entries are [priority_tuple, count, task]
        self._entry_finder = {}  # task -> entry
        self._counter = 0
        self.REMOVED = object()  # unique token

    def add_task(self, task: GridNode, priority: Tuple[float, float]):
        """Add or update a task's priority."""
        if task in self._entry_finder:
            self.remove_task(task)
        count = self._counter
        self._counter += 1
        entry = [priority, count, task]
        self._entry_finder[task] = entry
        heapq.heappush(self._queue, entry)

    def remove_task(self, task: GridNode):
        """Mark an existing task as removed. Lazy deletion."""
        entry = self._entry_finder.pop(task, None)
        if entry is not None:
            entry[2] = self.REMOVED

    def pop_task(self) -> Tuple[GridNode, Tuple[float, float]]:
        """Pop the lowest priority task. Raises KeyError if empty."""
        while self._queue:
            priority, count, task = heapq.heappop(self._queue)
            if task is not self.REMOVED:
                # real task
                try:
                    del self._entry_finder[task]
                except KeyError:
                    pass
                return task, priority
        raise KeyError('pop from an empty priority queue')

    def top_key(self) -> Optional[Tuple[float, float]]:
        """Return the key of the top element without popping or None if empty."""
        while self._queue and self._queue[0][2] is self.REMOVED:
            heapq.heappop(self._queue)
        return self._queue[0][0] if self._queue else None

    def empty(self) -> bool:
        return len(self._entry_finder) == 0


class DStarLite:
    """D* Lite incremental planner with fixes for dynamic obstacle handling."""

    def __init__(self, width: int, height: int, obstacles: Set[Tuple[int, int]] = None,
                 alpha: float = 1.0):
        self.width = width
        self.height = height
        self.obstacles = set(obstacles or set())  # static obstacles
        self.dynamic_obstacles: Set[Tuple[int, int]] = set()

        # Sparse maps for costs (optional)
        self.cell_costs: Dict[GridNode, float] = {}
        self.alpha = alpha

        # D* Lite bookkeeping
        self.start = GridNode(0, 0)
        self.goal = GridNode(0, 0)
        self.last_start = GridNode(0, 0)  # needed to compute km changes
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
        coord = (node.x, node.y)
        return coord in self.obstacles or coord in self.dynamic_obstacles

    def get_neighbors(self, node: GridNode) -> List[GridNode]:
        neighbors = []
        for motion in self.MOTIONS:
            nb = node + motion.to_node()
            if self.is_valid(nb):
                neighbors.append(nb)
        return neighbors

    def get_predecessors(self, node: GridNode) -> List[GridNode]:
        # On a grid predecessors == neighbors (undirected grid)
        return self.get_neighbors(node)

    def get_successors(self, node: GridNode) -> List[GridNode]:
        return self.get_neighbors(node)

    def cost(self, from_node: GridNode, to_node: GridNode) -> float:
        """Return cost of moving from 'from_node' to 'to_node'."""
        if self.is_obstacle(to_node):
            return math.inf

        diff = to_node - from_node
        for motion in self.MOTIONS:
            if motion.to_node() == diff:
                # incorporate optional occupancy cost
                cell_cost = self.cell_costs.get(to_node, 0.0)
                return edge_cost(cell_cost, motion.cost, alpha=self.alpha)
        return math.inf

    # ----------------------------- heuristics & keys -----------------------------
    def heuristic_between(self, a: GridNode, b: GridNode) -> float:
        return octile_heuristic(a, b)

    def heuristic(self, node: GridNode) -> float:
        # D* Lite uses h(node, start)
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
        """Return True if key1 < key2 lexicographically."""
        return key1[0] < key2[0] or (key1[0] == key2[0] and key1[1] < key2[1])

    # ----------------------------- core D* Lite ops -----------------------------
    def update_vertex(self, node: GridNode):
        """Update RHS value for a node and maintain priority queue consistency."""
        if node != self.goal:
            rhs_candidates = []
            for succ in self.get_successors(node):
                c = self.cost(node, succ)
                if c < math.inf:
                    rhs_candidates.append(c + self.get_g(succ))
            if rhs_candidates:
                self.set_rhs(node, min(rhs_candidates))
            else:
                self.set_rhs(node, math.inf)

        # maintain priority queue membership
        self.U.remove_task(node)
        if self.get_g(node) != self.get_rhs(node):
            self.U.add_task(node, self.calculate_key(node))

    def compute_shortest_path(self):
        """Main D* Lite computation loop."""
        iterations = 0
        
        while (not self.U.empty() and
               (self.compare_keys(self.U.top_key(), self.calculate_key(self.start)) or
                self.get_rhs(self.start) != self.get_g(self.start))):

            iterations += 1
            if iterations > 10000:  # Prevent infinite loops
                print("WARNING: compute_shortest_path exceeded iteration limit")
                break

            node, k_old = self.U.pop_task()
            k_new = self.calculate_key(node)

            if self.compare_keys(k_old, k_new):
                # key decreased/increased — reinsert
                self.U.add_task(node, k_new)
            elif self.get_g(node) > self.get_rhs(node):
                # node became overconsistent -> make consistent
                self.set_g(node, self.get_rhs(node))
                for pred in self.get_predecessors(node):
                    self.update_vertex(pred)
            else:
                # node is underconsistent -> set g to inf and update
                self.set_g(node, math.inf)
                # Update both predecessors AND the node itself
                for pred in self.get_predecessors(node):
                    self.update_vertex(pred)
                self.update_vertex(node)
        
        print(f"compute_shortest_path completed in {iterations} iterations")
        print(f"Start {self.start} g-value: {self.get_g(self.start)}, rhs: {self.get_rhs(self.start)}")

    # ----------------------------- initialization / external API -----------------------------
    def initialize(self, start: GridNode, goal: GridNode):
        """Initialize the planner for first-time use."""
        self.start = start
        self.goal = goal
        self.last_start = start
        
        self.initialized = True
        self.U = PriorityQueue()
        self.km = 0.0
        self.g_values.clear()
        self.rhs_values.clear()
        
        # Initialize goal
        self.set_rhs(self.goal, 0.0)
        self.U.add_task(self.goal, self.calculate_key(self.goal))

    def set_start(self, new_start: GridNode):
        """Move the robot start position (incremental replanning support)."""
        if not self.initialized:
            self.start = new_start
            return

        # Update km ONLY when start actually moves
        if new_start != self.start:
            self.km += self.heuristic_between(self.last_start, new_start)
            self.last_start = new_start
            self.start = new_start
            print(f"Start moved, km updated to {self.km}")

    def set_goal(self, new_goal: GridNode):
        """Change goal and reinitialize."""
        if new_goal != self.goal:
            # Clear previous goal
            old_goal = self.goal
            self.set_rhs(old_goal, math.inf)
            
            # Set new goal
            self.goal = new_goal
            self.set_rhs(self.goal, 0.0)
            self.U.add_task(self.goal, self.calculate_key(self.goal))

    def plan_path(self, start: GridNode, goal: GridNode) -> List[GridNode]:
        """Plan initial path or replan with new start/goal."""
        if not self.initialized:
            self.initialize(start, goal)
        else:
            # Update start/goal if changed
            if start != self.start:
                self.set_start(start)
            if goal != self.goal:
                self.set_goal(goal)

        self.compute_shortest_path()

        if self.get_g(self.start) == math.inf:
            print("No path exists from start to goal")
            return []
        
        return self.extract_path()

    def extract_path(self) -> List[GridNode]:
        """Extract path by following the gradient from start to goal."""
        if self.get_g(self.start) == math.inf:
            print("Cannot extract path: start is unreachable")
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

            # Find best successor based on actual cost + g-value
            best_succ = None
            best_total_cost = math.inf
            
            for succ in self.get_successors(current):
                if self.is_obstacle(succ) or succ in visited:
                    continue
                    
                edge_c = self.cost(current, succ)
                if edge_c == math.inf:
                    continue
                
                g_succ = self.get_g(succ)
                if g_succ == math.inf:
                    continue
                    
                total_cost = edge_c + g_succ
                
                if total_cost < best_total_cost:
                    best_total_cost = total_cost
                    best_succ = succ
            
            if best_succ is None:
                print(f"Path extraction failed at {current}")
                print(f"Current g: {self.get_g(current)}, rhs: {self.get_rhs(current)}")
                
                # Debug: show successor states
                for succ in self.get_successors(current):
                    if not self.is_obstacle(succ) and succ not in visited:
                        edge_c = self.cost(current, succ)
                        g_succ = self.get_g(succ)
                        print(f"  {succ}: edge_cost={edge_c}, g={g_succ}")
                
                return []
            
            current = best_succ

        if current == self.goal:
            path.append(self.goal)
            print(f"Successfully extracted path with {len(path)} nodes")
        
        return path

    def is_path_blocked(self, path: List[GridNode]) -> bool:
        """Check if any node in the path is now an obstacle."""
        for node in path:
            if self.is_obstacle(node):
                return True
        return False

    def update_dynamic_obstacles(self, added: Set[Tuple[int, int]], removed: Set[Tuple[int, int]]):
        """Update dynamic obstacles with proper D* Lite vertex updates."""
        
        if not added and not removed:
            return False
            
        print(f"Updating dynamic obstacles: +{len(added)}, -{len(removed)}")
        
        # Update obstacle sets
        for o in added:
            self.dynamic_obstacles.add(o)
        for o in removed:
            self.dynamic_obstacles.discard(o)
        
        # Collect all nodes that need updating
        nodes_to_update = set()
        
        # For each changed obstacle location
        for o in added | removed:
            node = GridNode(*o)
            if not self.is_valid(node):
                continue
                
            print(f"Processing changed obstacle at {node}")
            
            # The obstacle node itself needs updating
            nodes_to_update.add(node)
            
            # ALL neighbors of the obstacle node need updating because:
            # - If obstacle was added: neighbors can no longer use this node
            # - If obstacle was removed: neighbors might find better paths through this node
            for neighbor in self.get_neighbors(node):
                if self.is_valid(neighbor):
                    nodes_to_update.add(neighbor)
        
        print(f"Updating {len(nodes_to_update)} affected nodes")
        
        # Update all affected vertices using proper D* Lite update
        for node in nodes_to_update:
            self.update_vertex(node)
        
        # Recompute shortest path to propagate changes
        self.compute_shortest_path()
        return True


def visualize_grid(planner: DStarLite, path: List[GridNode] = None, figsize: Tuple[int, int] = (8, 8)):
    """Simple matplotlib visualization."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available - cannot visualize")
        return

    grid = np.zeros((planner.height, planner.width), dtype=np.uint8)

    # static obstacles
    for (x, y) in planner.obstacles:
        if 0 <= x < planner.width and 0 <= y < planner.height:
            grid[y, x] = 200

    # dynamic obstacles (different shade)
    for (x, y) in planner.dynamic_obstacles:
        if 0 <= x < planner.width and 0 <= y < planner.height:
            grid[y, x] = 120

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(grid, origin='lower', cmap='gray_r')

    if path:
        xs = [n.x for n in path]
        ys = [n.y for n in path]
        ax.plot(xs, ys, 'b-', linewidth=2, marker='o', markersize=4, alpha=0.8)

    # start & goal
    ax.scatter([planner.start.x], [planner.start.y], c='green', s=100, marker='*', label='start', zorder=5)
    ax.scatter([planner.goal.x], [planner.goal.y], c='red', s=100, marker='X', label='goal', zorder=5)
    ax.set_xlim(-0.5, planner.width - 0.5)
    ax.set_ylim(-0.5, planner.height - 0.5)
    ax.legend()
    ax.set_title('D* Lite Grid Planner')
    ax.grid(True, alpha=0.3)
    plt.show()


def demo_standalone():
    """Standalone demo with dynamic obstacle testing."""
    print("Running fixed D* Lite standalone demo...")
    width, height = 50, 50
    obstacles = {(20, y) for y in range(5, 25)}  # vertical wall
    planner = DStarLite(width, height, obstacles)

    start = GridNode(2, 10)
    goal = GridNode(38, 15)

    # Initial planning
    print("=== Initial Planning ===")
    path = planner.plan_path(start, goal)
    if path:
        print(f"Initial path length: {len(path)}")
        print(f"Path: {path[:5]}...{path[-5:] if len(path) > 10 else path[5:]}")
        visualize_grid(planner, path)

        # Test dynamic obstacles
        print("\n=== Dynamic Obstacle Testing ===")
        
        # Add obstacle that blocks the path - pick a node actually in the path
        print("Adding dynamic obstacle that blocks the path...")
        
        if len(path) > 10:
            mid_idx = len(path) // 2
            block_node = path[mid_idx]
            added = {(block_node.x, block_node.y)}
            print(f"Blocking path at node {mid_idx}: {block_node}")
        else:
            added = {(21, 10)}  # Block the passage around the wall
        
        print(f"Adding obstacle at: {added}")
        
        planner.update_dynamic_obstacles(added, set())
        print(f"Path blocked: {planner.is_path_blocked(path)}")
        
        new_path = planner.extract_path()
        if new_path:
            print(f"Replanned path length: {len(new_path)}")
            print(f"New path: {new_path[:5]}...{new_path[-5:] if len(new_path) > 10 else new_path[5:]}")
            visualize_grid(planner, new_path)
        else:
            print("No path found after adding dynamic obstacle")
            
        # Test removing the obstacle
        print("\nRemoving dynamic obstacle...")
        planner.update_dynamic_obstacles(set(), added)
        restored_path = planner.extract_path()
        if restored_path:
            print(f"Restored path length: {len(restored_path)}")
            visualize_grid(planner, restored_path)
        else:
            print("Still no path after removing obstacle")
            
    else:
        print("No initial path found")


if __name__ == '__main__':
    demo_standalone()