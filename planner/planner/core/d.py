#!/usr/bin/env python3
"""
Improved D* Lite grid planner (cleaned + fixes + visualization)

Changes from your original version (high level):
- Fixed ROS2 import detection flag
- Properly initialize motion set
- Robust priority queue with unique REMOVED token
- Correct handling of km when the start moves
- Separate static vs dynamic obstacles and clearer update paths
- Optional occupancy costs and octile heuristic
- Added visualization helper (matplotlib) for standalone/demo use
- Added `set_start` / `set_goal` to support incremental replanning
- FIXED: Dynamic obstacle updates following working D* Lite pattern

This file is intended as drop-in replacement for the structure you provided.
"""

import heapq
import math
from typing import List, Tuple, Set, Optional, Dict
from dataclasses import dataclass
import numpy as np
import time

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Point, PoseStamped
    from nav_msgs.msg import OccupancyGrid, Path
    from std_msgs.msg import Header
    from visualization_msgs.msg import Marker, MarkerArray
    ROS2_AVAILABLE = True
except Exception:
    # Running standalone mode
    print("ROS2 not available, running in standalone mode")
    ROS2_AVAILABLE = False

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
    """D* Lite incremental planner with some practical fixes and helpers."""

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
        iterations = 0
        # main loop
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
                # node became consistent
                self.set_g(node, self.get_rhs(node))
                for pred in self.get_predecessors(node):
                    self.update_vertex(pred)
            else:
                # node is underconsistent -> set to inf and update preds
                self.set_g(node, math.inf)
                for pred in self.get_predecessors(node) + [node]:
                    self.update_vertex(pred)
        
        print(f"compute_shortest_path completed in {iterations} iterations")
        print(f"Start g-value: {self.get_g(self.start)}, rhs: {self.get_rhs(self.start)}")

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
            # keep static obstacles as-is; do not clear

            self.set_rhs(self.goal, 0.0)
            self.U.add_task(self.goal, self.calculate_key(self.goal))

    def set_start(self, new_start: GridNode):
        """Move the robot start position (incremental replanning support)."""
        if not self.initialized:
            # if not initialized, just set start; plan_path will initialize
            self.start = new_start
            return

        # Update km by the distance between last known start and new
        self.km += self.heuristic_between(self.last_start, new_start)
        self.last_start = new_start
        self.start = new_start

        # Update affected area (start moved) — update neighbors
        self.update_vertex(self.start)
        self.compute_shortest_path()

    def set_goal(self, new_goal: GridNode):
        # change goal and reset rhs for the new goal
        self.goal = new_goal
        self.set_rhs(self.goal, 0.0)
        self.U.add_task(self.goal, self.calculate_key(self.goal))
        self.compute_shortest_path()

    def plan_path(self, start: GridNode, goal: GridNode) -> List[GridNode]:
        # initialize only if first time
        if not self.initialized:
            self.initialize(start, goal)
        else:
            # if planner already initialized and start/goal change, use set_start/set_goal
            if start != self.start:
                self.set_start(start)
            if goal != self.goal:
                self.set_goal(goal)

        self.compute_shortest_path()

        if self.get_g(self.start) == math.inf:
            return []
        return self.extract_path()

    def extract_path(self) -> List[GridNode]:
        """Fixed path extraction that handles dynamic changes properly."""
        if self.get_g(self.start) == math.inf:
            return []

        path: List[GridNode] = []
        current = self.start
        visited = set()
        max_iterations = self.width * self.height  # Prevent infinite loops

        iteration_count = 0
        while current != self.goal and iteration_count < max_iterations:
            path.append(current)
            visited.add(current)
            iteration_count += 1

            best_succ = None
            best_cost = math.inf

            for succ in self.get_successors(current):
                # Skip if obstacle or already visited
                if self.is_obstacle(succ) or succ in visited:
                    continue
                    
                c = self.cost(current, succ)
                if c == math.inf:
                    continue
                    
                # CRITICAL: Use current g-values, not cached ones
                g_succ = self.get_g(succ)
                val = c + g_succ
                
                if val < best_cost:
                    best_cost = val
                    best_succ = succ

            if best_succ is None:
                # No valid successor found - path is blocked
                print(f"Path blocked at {current}, triggering emergency replan")
                
                # Force a complete recomputation
                self.km += self.heuristic_between(self.last_start, current)
                self.last_start = current
                self.start = current
                
                # Update the current node and recompute
                self.update_vertex(current)
                self.compute_shortest_path()
                
                # If still no path, return partial path
                if self.get_g(current) == math.inf:
                    return path
                    
                # Try again with updated values
                continue
                
            current = best_succ

        if current == self.goal:
            path.append(self.goal)
        
        return path

    def update_static_obstacles(self, new_static: Set[Tuple[int, int]]):
        """Replace static obstacles and trigger minimal replan."""
        old = self.obstacles
        self.obstacles = set(new_static)

        new = self.obstacles - old
        removed = old - self.obstacles
        changed_nodes = set()

        for (x, y) in new | removed:
            n = GridNode(x, y)
            if self.is_valid(n):
                changed_nodes.add(n)

        if not changed_nodes:
            return False

        # update neighbors of changed nodes
        for n in changed_nodes:
            for nb in self.get_neighbors(n) + [n]:
                if self.is_valid(nb):
                    self.update_vertex(nb)

        self.compute_shortest_path()
        return True

    def is_path_blocked(self, path: List[GridNode]) -> bool:
        """Check if any node in the path is now an obstacle."""
        for node in path:
            if self.is_obstacle(node):
                return True
        return False

    def update_dynamic_obstacles(self, added: Set[Tuple[int, int]], removed: Set[Tuple[int, int]]):
        """Fixed version following the working D* Lite pattern with improved heuristics and heap."""
        changed_nodes = set()
        
        # Process added obstacles
        for o in added:
            if o not in self.dynamic_obstacles:
                self.dynamic_obstacles.add(o)
                n = GridNode(o[0], o[1])
                if self.is_valid(n):
                    changed_nodes.add(n)
        
        # Process removed obstacles  
        for o in removed:
            if o in self.dynamic_obstacles:
                self.dynamic_obstacles.remove(o)
                n = GridNode(o[0], o[1])
                if self.is_valid(n):
                    changed_nodes.add(n)

        if not changed_nodes:
            return False

        # Update km similar to working implementation when obstacles change
        if changed_nodes:
            self.km += self.heuristic_between(self.last_start, self.start)
            self.last_start = self.start

        # CRITICAL: Follow the working implementation pattern
        # Directly invalidate obstacle nodes, then update them
        for node in changed_nodes:
            if node == self.start:  # Don't invalidate start
                continue
                
            if self.is_obstacle(node):
                # Node became obstacle: set g and rhs to infinity (like working version)
                self.set_g(node, math.inf)
                self.set_rhs(node, math.inf)
            # For removed obstacles, don't set values - let update_vertex recalculate
                
            # Update the changed vertex itself (key step from working version)
            self.update_vertex(node)

        # Run compute_shortest_path to propagate changes
        self.compute_shortest_path()
        return True


# ------------------------- ROS2 wrapper (keeps your structure) -------------------------
# class DStarLiteROS2Node(Node):
#     def __init__(self):
#         super().__init__('dstar_lite_planner')
#         self.declare_parameter('grid_width', 100)
#         self.declare_parameter('grid_height', 100)
#         self.declare_parameter('resolution', 0.1)
#         self.declare_parameter('origin_x', 0.0)
#         self.declare_parameter('origin_y', 0.0)

#         self.grid_width = self.get_parameter('grid_width').value
#         self.grid_height = self.get_parameter('grid_height').value
#         self.resolution = self.get_parameter('resolution').value
#         self.origin_x = self.get_parameter('origin_x').value
#         self.origin_y = self.get_parameter('origin_y').value

#         self.planner = DStarLite(self.grid_width, self.grid_height)

#         self.path_pub = self.create_publisher(Path, 'planned_path', 10)
#         self.marker_pub = self.create_publisher(MarkerArray, 'path_markers', 10)

#         self.map_sub = self.create_subscription(OccupancyGrid, 'map', self.map_callback, 10)
#         self.goal_sub = self.create_subscription(PoseStamped, 'goal_pose', self.goal_callback, 10)
#         self.start_sub = self.create_subscription(PoseStamped, 'start_pose', self.start_callback, 10)

#         self.current_start: Optional[GridNode] = None
#         self.current_goal: Optional[GridNode] = None

#         self.get_logger().info('D* Lite planner node initialized')

#     def world_to_grid(self, x: float, y: float) -> GridNode:
#         gx = int((x - self.origin_x) / self.resolution)
#         gy = int((y - self.origin_y) / self.resolution)
#         return GridNode(gx, gy)

#     def grid_to_world(self, node: GridNode) -> Tuple[float, float]:
#         wx = node.x * self.resolution + self.origin_x
#         wy = node.y * self.resolution + self.origin_y
#         return wx, wy

#     def map_callback(self, msg: OccupancyGrid):
#         obstacles = set()
#         width = msg.info.width
#         height = msg.info.height
#         data = msg.data
#         arr = np.frombuffer(bytearray(data), dtype=np.int8) if isinstance(data, (bytes, bytearray)) else np.array(data, dtype=np.int8)
#         arr = arr.reshape((height, width))

#         # threshold as before
#         ys, xs = np.where(arr > 50)
#         for x, y in zip(xs, ys):
#             obstacles.add((int(x), int(y)))

#         old = self.planner.obstacles.copy()
#         self.planner.obstacles = obstacles
#         new_obstacles = obstacles - old
#         removed_obstacles = old - obstacles

#         if self.planner.initialized and (new_obstacles or removed_obstacles):
#             self.planner.update_static_obstacles(obstacles)
#             self.replan_and_publish()

#     def start_callback(self, msg: PoseStamped):
#         s = self.world_to_grid(msg.pose.position.x, msg.pose.position.y)
#         self.current_start = s
#         if not self.planner.initialized:
#             # first plan will initialize the planner
#             self.plan_if_ready()
#         else:
#             # incremental update
#             self.planner.set_start(s)
#             self.replan_and_publish()

#     def goal_callback(self, msg: PoseStamped):
#         g = self.world_to_grid(msg.pose.position.x, msg.pose.position.y)
#         self.current_goal = g
#         if not self.planner.initialized:
#             self.plan_if_ready()
#         else:
#             self.planner.set_goal(g)
#             self.replan_and_publish()

#     def plan_if_ready(self):
#         if self.current_start is not None and self.current_goal is not None:
#             path = self.planner.plan_path(self.current_start, self.current_goal)
#             self.publish_path(path)

#     def replan_and_publish(self):
#         if self.current_start is not None and self.current_goal is not None and self.planner.initialized:
#             path = self.planner.extract_path()
#             self.publish_path(path)

#     def publish_path(self, path: List[GridNode]):
#         if not path:
#             self.get_logger().warn('No valid path found')
#             return
#         path_msg = Path()
#         path_msg.header = Header()
#         path_msg.header.stamp = self.get_clock().now().to_msg()
#         path_msg.header.frame_id = 'map'

#         for node in path:
#             ps = PoseStamped()
#             ps.header = path_msg.header
#             wx, wy = self.grid_to_world(node)
#             ps.pose.position.x = wx
#             ps.pose.position.y = wy
#             ps.pose.position.z = 0.0
#             ps.pose.orientation.w = 1.0
#             path_msg.poses.append(ps)

#         self.path_pub.publish(path_msg)
#         self.get_logger().info(f'Published path with {len(path)} waypoints')


# ------------------------- Visualization & demo helpers -------------------------
def visualize_grid(planner: DStarLite, path: List[GridNode] = None, figsize: Tuple[int, int] = (8, 8)):
    """Simple matplotlib visualization: obstacles, dynamic obstacles, start, goal, and path."""
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
        ax.plot(xs, ys, linewidth=2, marker='o')

    # start & goal
    ax.scatter([planner.start.x], [planner.start.y], c='green', s=100, marker='*', label='start')
    ax.scatter([planner.goal.x], [planner.goal.y], c='red', s=100, marker='X', label='goal')
    ax.set_xlim(-0.5, planner.width - 0.5)
    ax.set_ylim(-0.5, planner.height - 0.5)
    ax.legend()
    ax.set_title('D* Lite grid')
    plt.show()


# ------------------------- standalone demo -------------------------

def demo_standalone():
    print("Running improved D* Lite standalone demo...")
    width, height = 50, 50
    obstacles = {(20, y) for y in range(5, 25)}  # vertical wall
    planner = DStarLite(width, height, obstacles)

    start = GridNode(2, 10)
    goal = GridNode(38, 15)

    path = planner.plan_path(start, goal)
    if path:
        print(f"Initial path length: {len(path)}")
        visualize_grid(planner, path)

        for i in range (5): ## 5 iterations of dynamic obstacle addition
            # Add dynamic obstacle that blocks path and replan
            added = {(10+i, 6+i), (int(11*(i/3)), 2), (int(12+2*i), 10-i)}
            print(f"Adding dynamic obstacles, replanning...{added}")
            planner.update_dynamic_obstacles(added, set())
            new_path = planner.extract_path()
            if new_path:
                print(f"Replanned path length: {len(new_path)}")
                visualize_grid(planner, new_path)
            else:
                print("No path after dynamic obstacles")
    else:
        print("No initial path found")


def main(args=None):
    if not ROS2_AVAILABLE:
        demo_standalone()
        return

    # rclpy.init(args=args)
    # node = DStarLiteROS2Node()
    # try:
    #     rclpy.spin(node)
    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     node.destroy_node()
    #     rclpy.shutdown()


if __name__ == '__main__':
    main()