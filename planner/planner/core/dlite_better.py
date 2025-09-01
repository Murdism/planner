#!/usr/bin/env python3
"""
Improved D* Lite grid planning with ROS2 integration
Optimizations:
- Proper priority queue implementation
- Efficient data structures
- ROS2 compatibility
- Better performance and memory usage
- Cleaner code structure
"""

import heapq
import math
from typing import List, Tuple, Set, Optional, Dict
from dataclasses import dataclass
import numpy as np
import time

# ROS2 imports
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Point, PoseStamped
    from nav_msgs.msg import OccupancyGrid, Path
    from std_msgs.msg import Header
    from visualization_msgs.msg import Marker, MarkerArray
    ROS2_AVAILABLE = False
except ImportError:
    print("ROS2 not available, running in standalone mode")
    ROS2_AVAILABLE = False

@dataclass(frozen=True)
class GridNode:
    """Immutable grid node for hashing in sets/dicts"""
    x: int
    y: int
    
    def __add__(self, other):
        return GridNode(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return GridNode(self.x - other.x, self.y - other.y)

@dataclass
class Motion:
    """Motion primitive with cost"""
    dx: int
    dy: int
    cost: float
    
    def to_node(self) -> GridNode:
        return GridNode(self.dx, self.dy)
    
def motions_grid8() -> List[Motion]:
    """8-connected global planner motions (orientation-free)."""
    card = [(1,0), (-1,0), (0,1), (0,-1)]
    diag = [(1,1), (1,-1), (-1,1), (-1,-1)]
    return [Motion(dx, dy, 1.0) for dx,dy in card] + \
        [Motion(dx, dy, math.sqrt(2.0)) for dx,dy in diag]

def octile_heuristic(a: GridNode, b: GridNode) -> float:
    dx = abs(a.x - b.x)
    dy = abs(a.y - b.y)
    dmax, dmin = (dx, dy) if dx >= dy else (dy, dx)
    return dmax + (math.sqrt(2.0) - 1.0) * dmin

def edge_cost(cell_cost: float, geo_step: float, alpha: float = 1.0) -> float:
    """
    cell_cost: normalized 0..1 (map cost or inflation)
    geo_step: 1.0 for straight, sqrt(2) for diagonal
    """
    return geo_step * (1.0 + alpha * cell_cost)

class PriorityQueue:
    """Efficient priority queue for D* Lite"""
    
    def __init__(self):
        self._queue = []
        self._entry_finder = {}
        self._counter = 0
        self.REMOVED = '<removed-task>'
    
    def add_task(self, task: GridNode, priority: Tuple[float, float]):
        """Add a new task or update the priority of an existing task"""
        if task in self._entry_finder:
            self.remove_task(task)
        count = self._counter
        self._counter += 1
        entry = [priority, count, task]
        self._entry_finder[task] = entry
        heapq.heappush(self._queue, entry)
    
    def remove_task(self, task: GridNode):
        """Mark an existing task as REMOVED"""
        if task not in self._entry_finder:
            return
        entry = self._entry_finder.pop(task)
        entry[-1] = self.REMOVED
    
    def pop_task(self) -> Tuple[GridNode, Tuple[float, float]]:
        """Remove and return the lowest priority task"""
        while self._queue:
            priority, count, task = heapq.heappop(self._queue)
            if task is not self.REMOVED:
                del self._entry_finder[task]
                return task, priority
        raise KeyError('pop from an empty priority queue')
    
    def top_key(self) -> Optional[Tuple[float, float]]:
        """Return the top priority without popping"""
        while self._queue and self._queue[0][-1] is self.REMOVED:
            heapq.heappop(self._queue)
        return self._queue[0][0] if self._queue else None
    
    def empty(self) -> bool:
        return len(self._entry_finder) == 0

class DStarLite:
    """Improved D* Lite implementation with optimizations"""

    def __init__(self, width: int, height: int, obstacles: Set[Tuple[int, int]] = None):
        """
        Initialize D* Lite planner
        
        Args:
            width: Grid width
            height: Grid height  
            obstacles: Set of obstacle coordinates (x, y)
        """
        self.width = width
        self.height = height
        self.obstacles = obstacles or set()
        self.dynamic_obstacles = set()
        
        # D* Lite variables
        self.start = GridNode(0, 0)
        self.goal = GridNode(0, 0)
        self.km = 0.0
        
        # Use dictionaries for sparse representation (memory efficient)
        self.g_values: Dict[GridNode, float] = {}
        self.rhs_values: Dict[GridNode, float] = {}
        self.U = PriorityQueue()
        
        self.initialized = False
        
    def get_g(self, node: GridNode) -> float:
        """Get g-value with default infinity"""
        return self.g_values.get(node, math.inf)
    
    def set_g(self, node: GridNode, value: float):
        """Set g-value"""
        if value == math.inf and node in self.g_values:
            del self.g_values[node]
        else:
            self.g_values[node] = value
    
    def get_rhs(self, node: GridNode) -> float:
        """Get rhs-value with default infinity"""
        return self.rhs_values.get(node, math.inf)
    
    def set_rhs(self, node: GridNode, value: float):
        """Set rhs-value"""
        if value == math.inf and node in self.rhs_values:
            del self.rhs_values[node]
        else:
            self.rhs_values[node] = value
    
    def is_valid(self, node: GridNode) -> bool:
        """Check if node is within grid bounds"""
        return 0 <= node.x < self.width and 0 <= node.y < self.height
    
    def is_obstacle(self, node: GridNode) -> bool:
        """Check if node is an obstacle"""
        coord = (node.x, node.y)
        return coord in self.obstacles or coord in self.dynamic_obstacles
    
    def get_neighbors(self, node: GridNode) -> List[GridNode]:
        """Get valid neighbors of a node"""
        neighbors = []
        for motion in self.MOTIONS:
            neighbor = node + motion.to_node()
            if self.is_valid(neighbor):
                neighbors.append(neighbor)
        return neighbors
    
    def get_predecessors(self, node: GridNode) -> List[GridNode]:
        """Get predecessors (same as neighbors in grid)"""
        return self.get_neighbors(node)
    
    def get_successors(self, node: GridNode) -> List[GridNode]:
        """Get successors (same as neighbors in grid)"""  
        return self.get_neighbors(node)
    
    def cost(self, from_node: GridNode, to_node: GridNode) -> float:
        """Get cost between two adjacent nodes"""
        if self.is_obstacle(to_node):
            return math.inf
        
        # Find the motion that matches this transition
        diff = to_node - from_node
        for motion in self.MOTIONS:
            if motion.to_node() == diff:
                return motion.cost
        
        return math.inf  # Not a valid transition
    
    def heuristic(self, node: GridNode) -> float:
        """Heuristic function (Euclidean distance)"""
        return math.sqrt((node.x - self.start.x)**2 + (node.y - self.start.y)**2)
    
    def calculate_key(self, node: GridNode) -> Tuple[float, float]:
        """Calculate priority key for a node"""
        g_val = self.get_g(node)
        rhs_val = self.get_rhs(node)
        min_val = min(g_val, rhs_val)
        
        k1 = min_val + self.heuristic(node) + self.km
        k2 = min_val
        
        return (k1, k2)
    
    def compare_keys(self, key1: Tuple[float, float], key2: Tuple[float, float]) -> bool:
        """Compare two keys (key1 < key2)"""
        return key1[0] < key2[0] or (key1[0] == key2[0] and key1[1] < key2[1])
    
    def update_vertex(self, node: GridNode):
        """Update vertex according to D* Lite algorithm"""
        # Update rhs value
        if node != self.goal:
            rhs_candidates = []
            for successor in self.get_successors(node):
                cost_val = self.cost(node, successor)
                if cost_val < math.inf:
                    rhs_candidates.append(cost_val + self.get_g(successor))
            
            if rhs_candidates:
                self.set_rhs(node, min(rhs_candidates))
            else:
                self.set_rhs(node, math.inf)
        
        # Update priority queue
        self.U.remove_task(node)
        
        if self.get_g(node) != self.get_rhs(node):
            self.U.add_task(node, self.calculate_key(node))
    
    def compute_shortest_path(self):
        """Main D* Lite computation"""
        while (not self.U.empty() and 
               (self.compare_keys(self.U.top_key(), self.calculate_key(self.start)) or
                self.get_rhs(self.start) != self.get_g(self.start))):
            
            node, k_old = self.U.pop_task()
            k_new = self.calculate_key(node)
            
            if self.compare_keys(k_old, k_new):
                # Key has changed, reinsert with new key
                self.U.add_task(node, k_new)
            elif self.get_g(node) > self.get_rhs(node):
                # Overconsistent case
                self.set_g(node, self.get_rhs(node))
                for pred in self.get_predecessors(node):
                    self.update_vertex(pred)
            else:
                # Underconsistent case  
                self.set_g(node, math.inf)
                for pred in self.get_predecessors(node) + [node]:
                    self.update_vertex(pred)
    
    def initialize(self, start: GridNode, goal: GridNode):
        """Initialize D* Lite"""
        self.start = start
        self.goal = goal
        
        if not self.initialized:
            self.initialized = True
            self.U = PriorityQueue()
            self.km = 0.0
            self.g_values.clear()
            self.rhs_values.clear()
            self.dynamic_obstacles.clear()
            
            # Initialize goal
            self.set_rhs(self.goal, 0.0)
            self.U.add_task(self.goal, self.calculate_key(self.goal))
    
    def plan_path(self, start: GridNode, goal: GridNode) -> List[GridNode]:
        """Plan initial path"""
        self.initialize(start, goal)
        self.compute_shortest_path()
        
        if self.get_g(self.start) == math.inf:
            return []  # No path found
        
        return self.extract_path()
    
    def extract_path(self) -> List[GridNode]:
        """Extract path from start to goal"""
        if self.get_g(self.start) == math.inf:
            return []
        
        path = []
        current = self.start
        
        while current != self.goal:
            path.append(current)
            
            # Find best successor
            best_successor = None
            best_cost = math.inf
            
            for successor in self.get_successors(current):
                total_cost = self.cost(current, successor) + self.get_g(successor)
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_successor = successor
            
            if best_successor is None:
                break  # No valid path
            
            current = best_successor
        
        path.append(self.goal)
        return path
    
    def update_obstacles(self, new_obstacles: Set[Tuple[int, int]], 
                        removed_obstacles: Set[Tuple[int, int]] = None) -> bool:
        """
        Update dynamic obstacles and replan if needed
        
        Returns:
            True if replanning was performed
        """
        removed_obstacles = removed_obstacles or set()
        
        # Track changes
        changed_nodes = set()
        
        # Add new obstacles
        for obs in new_obstacles:
            if obs not in self.dynamic_obstacles:
                self.dynamic_obstacles.add(obs)
                node = GridNode(obs[0], obs[1])
                if self.is_valid(node):
                    changed_nodes.add(node)
        
        # Remove obstacles
        for obs in removed_obstacles:
            if obs in self.dynamic_obstacles:
                self.dynamic_obstacles.remove(obs)
                node = GridNode(obs[0], obs[1])
                if self.is_valid(node):
                    changed_nodes.add(node)
        
        if not changed_nodes:
            return False
        
        # Update affected vertices
        last_start = self.start
        self.km += self.heuristic(last_start)
        
        for node in changed_nodes:
            # Update the changed node and its neighbors
            for neighbor in self.get_neighbors(node) + [node]:
                if self.is_valid(neighbor):
                    self.update_vertex(neighbor)
        
        # Recompute shortest path
        self.compute_shortest_path()
        return True

class DStarLiteROS2Node(Node):
    """ROS2 node wrapper for D* Lite planner"""
    
    def __init__(self):
        super().__init__('dstar_lite_planner')
        
        # Parameters
        self.declare_parameter('grid_width', 100)
        self.declare_parameter('grid_height', 100)
        self.declare_parameter('resolution', 0.1)  # meters per cell
        self.declare_parameter('origin_x', 0.0)
        self.declare_parameter('origin_y', 0.0)
        
        self.grid_width = self.get_parameter('grid_width').value
        self.grid_height = self.get_parameter('grid_height').value
        self.resolution = self.get_parameter('resolution').value
        self.origin_x = self.get_parameter('origin_x').value
        self.origin_y = self.get_parameter('origin_y').value
        
        # Initialize planner
        self.planner = DStarLite(self.grid_width, self.grid_height)
        
        # ROS2 publishers and subscribers
        self.path_pub = self.create_publisher(Path, 'planned_path', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'path_markers', 10)
        
        self.map_sub = self.create_subscription(
            OccupancyGrid, 'map', self.map_callback, 10)
        self.goal_sub = self.create_subscription(
            PoseStamped, 'goal_pose', self.goal_callback, 10)
        self.start_sub = self.create_subscription(
            PoseStamped, 'start_pose', self.start_callback, 10)
        
        self.current_start = None
        self.current_goal = None
        
        self.get_logger().info('D* Lite planner node initialized')
    
    def world_to_grid(self, x: float, y: float) -> GridNode:
        """Convert world coordinates to grid coordinates"""
        grid_x = int((x - self.origin_x) / self.resolution)
        grid_y = int((y - self.origin_y) / self.resolution)
        return GridNode(grid_x, grid_y)
    
    def grid_to_world(self, node: GridNode) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates"""
        world_x = node.x * self.resolution + self.origin_x
        world_y = node.y * self.resolution + self.origin_y
        return world_x, world_y
    
    def map_callback(self, msg: OccupancyGrid):
        """Handle incoming occupancy grid map"""
        obstacles = set()
        
        width = msg.info.width
        height = msg.info.height
        
        for y in range(height):
            for x in range(width):
                index = y * width + x
                if index < len(msg.data) and msg.data[index] > 50:  # Occupied
                    obstacles.add((x, y))
        
        # Update planner with new obstacles
        old_obstacles = self.planner.obstacles.copy()
        self.planner.obstacles = obstacles
        
        new_obstacles = obstacles - old_obstacles
        removed_obstacles = old_obstacles - obstacles
        
        if new_obstacles or removed_obstacles:
            if self.planner.initialized:
                self.planner.update_obstacles(new_obstacles, removed_obstacles)
                self.replan_and_publish()
    
    def start_callback(self, msg: PoseStamped):
        """Handle start pose updates"""
        start_grid = self.world_to_grid(msg.pose.position.x, msg.pose.position.y)
        self.current_start = start_grid
        self.plan_if_ready()
    
    def goal_callback(self, msg: PoseStamped):
        """Handle goal pose updates"""  
        goal_grid = self.world_to_grid(msg.pose.position.x, msg.pose.position.y)
        self.current_goal = goal_grid
        self.plan_if_ready()
    
    def plan_if_ready(self):
        """Plan path if both start and goal are set"""
        if self.current_start and self.current_goal:
            path = self.planner.plan_path(self.current_start, self.current_goal)
            self.publish_path(path)
    
    def replan_and_publish(self):
        """Replan and publish updated path"""
        if self.current_start and self.current_goal and self.planner.initialized:
            path = self.planner.extract_path()
            self.publish_path(path)
    
    def publish_path(self, path: List[GridNode]):
        """Publish path as ROS2 Path message"""
        if not path:
            self.get_logger().warn('No valid path found')
            return
        
        # Create Path message
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'
        
        for node in path:
            pose_stamped = PoseStamped()
            pose_stamped.header = path_msg.header
            
            world_x, world_y = self.grid_to_world(node)
            pose_stamped.pose.position.x = world_x
            pose_stamped.pose.position.y = world_y
            pose_stamped.pose.position.z = 0.0
            pose_stamped.pose.orientation.w = 1.0
            
            path_msg.poses.append(pose_stamped)
        
        self.path_pub.publish(path_msg)
        self.get_logger().info(f'Published path with {len(path)} waypoints')

def main(args=None):
    """Main function for ROS2 node"""
    if not ROS2_AVAILABLE:
        print("ROS2 not available. Running standalone demo...")
        # Run standalone demo
        demo_standalone()
        return
    
    rclpy.init(args=args)
    node = DStarLiteROS2Node()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

def demo_standalone():
    """Standalone demo without ROS2"""
    print("Running D* Lite standalone demo...")
    
    # Create a simple test scenario
    width, height = 20, 20
    obstacles = {(10, i) for i in range(5, 15)}  # Vertical wall
    
    planner = DStarLite(width, height, obstacles)
    
    start = GridNode(2, 10)
    goal = GridNode(18, 10)
    
    print(f"Planning from {start} to {goal}")
    path = planner.plan_path(start, goal)
    
    if path:
        print(f"Initial path found with {len(path)} nodes")
        print("Path:", [f"({n.x},{n.y})" for n in path])
        
        # Add dynamic obstacle
        print("\nAdding dynamic obstacle...")
        new_obstacles = {(5, 10), (6, 10), (7, 10)}
        planner.update_obstacles(new_obstacles)
        
        new_path = planner.extract_path()
        if new_path:
            print(f"Replanned path with {len(new_path)} nodes")
            print("New path:", [f"({n.x},{n.y})" for n in new_path])
        else:
            print("No path found after obstacle addition")
    else:
        print("No initial path found")

if __name__ == '__main__':
    main()