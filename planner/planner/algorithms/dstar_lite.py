#!/usr/bin/env python3
import heapq
import copy
import math
from typing import List, Tuple, Set, Optional, Dict, NamedTuple
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

    def __hash__(self):
        return hash((self.x, self.y))


class Motion(NamedTuple):
    """Immutable motion with efficient hashing and comparison."""
    dx: int
    dy: int
    cost: float
    
    def to_node(self) -> GridNode:
        return GridNode(self.dx, self.dy)
    
    @property
    def is_diagonal(self) -> bool:
        return abs(self.dx) == abs(self.dy) == 1


# Pre-computed motion sets for better performance
_CARDINAL_MOTIONS = [
    Motion(1, 0, 1.0),   Motion(-1, 0, 1.0),
    Motion(0, 1, 1.0),   Motion(0, -1, 1.0)
]

_DIAGONAL_MOTIONS = [
    Motion(1, 1, math.sqrt(2.0)),   Motion(1, -1, math.sqrt(2.0)),
    Motion(-1, 1, math.sqrt(2.0)),  Motion(-1, -1, math.sqrt(2.0))
]

_GRID8_MOTIONS = _CARDINAL_MOTIONS + _DIAGONAL_MOTIONS

# Create lookup table for fast motion retrieval
_MOTION_LOOKUP = {(m.dx, m.dy): m for m in _GRID8_MOTIONS}


def motions_grid8() -> List[Motion]:
    """8-connected grid motions (cardinal + diagonal) - cached for performance."""
    return _GRID8_MOTIONS.copy()


def get_motion_for_diff(dx: int, dy: int) -> Optional[Motion]:
    """Fast lookup for motion given coordinate difference."""
    return _MOTION_LOOKUP.get((dx, dy))


def octile_heuristic(a: GridNode, b: GridNode) -> float:
    """Optimized octile distance heuristic."""
    dx = abs(a.x - b.x)
    dy = abs(a.y - b.y)
    if dx >= dy:
        return dx + (math.sqrt(2.0) - 1.0) * dy
    else:
        return dy + (math.sqrt(2.0) - 1.0) * dx


def edge_cost(cell_cost: float, geo_step: float, alpha: float = 1.0) -> float:
    """
    Convert a normalized cell cost (0..1) into an edge cost.
    geo_step: 1.0 for cardinal, sqrt(2) for diagonal.
    alpha: controls how strongly the cell cost affects the edge.
    """
    return geo_step * (1.0 + alpha * cell_cost)


class AdvancedPriorityQueue:
    """
    Enhanced priority queue with O(log n) updates and lazy deletion.
    Uses binary heap with entry tracking for efficient priority updates.
    """

    def __init__(self):
        self._heap = []  # [priority_tuple, counter, task]
        self._task_to_entry = {}  # task -> heap_entry
        self._counter = 0
        self.REMOVED = object()  # sentinel for removed entries

    def add_or_update_task(self, task: GridNode, priority: Tuple[float, float]):
        """Add new task or update existing task's priority in O(log n)."""
        # Remove old entry if exists
        if task in self._task_to_entry:
            old_entry = self._task_to_entry[task]
            old_entry[2] = self.REMOVED  # mark as removed

        # Add new entry
        count = self._counter
        self._counter += 1
        entry = [priority, count, task]
        self._task_to_entry[task] = entry
        heapq.heappush(self._heap, entry)

    def remove_task(self, task: GridNode) -> bool:
        """Mark task as removed (lazy deletion). Returns True if task existed."""
        entry = self._task_to_entry.pop(task, None)
        if entry is not None:
            entry[2] = self.REMOVED
            return True
        return False

    def pop_task(self) -> Tuple[GridNode, Tuple[float, float]]:
        """Pop lowest priority task. Raises KeyError if empty."""
        while self._heap:
            priority, count, task = heapq.heappop(self._heap)
            if task is not self.REMOVED:
                self._task_to_entry.pop(task, None)
                return task, priority
        raise KeyError('pop from empty priority queue')

    def top_key(self) -> Optional[Tuple[float, float]]:
        """Return priority of top element without popping, or None if empty."""
        self._cleanup_top()
        return self._heap[0][0] if self._heap else None

    def _cleanup_top(self):
        """Remove removed entries from top of heap."""
        while self._heap and self._heap[0][2] is self.REMOVED:
            heapq.heappop(self._heap)

    def empty(self) -> bool:
        """Return True if queue is empty."""
        return len(self._task_to_entry) == 0

    def size(self) -> int:
        """Return number of active tasks."""
        return len(self._task_to_entry)

    def contains(self, task: GridNode) -> bool:
        """Return True if task is in queue."""
        return task in self._task_to_entry


class DStarLite:
    """Enhanced D* Lite incremental planner with improved heap and motion handling."""

    def __init__(self, width: int, height: int, obstacles: Set[Tuple[int, int]] = None,
                 alpha: float = 1.0):
        self.width = width
        self.height = height
        self.obstacles = set(obstacles or set())  # static obstacles
        self.dynamic_obstacles: Set[Tuple[int, int]] = set()

        # Sparse maps for costs
        self.cell_costs: Dict[GridNode, float] = {}
        self.alpha = alpha

        # D* Lite state
        self.start = GridNode(0, 0)
        self.goal = GridNode(0, 0)
        self.last_start = GridNode(0, 0)
        self.km = 0.0

        self.g_values: Dict[GridNode, float] = {}
        self.rhs_values: Dict[GridNode, float] = {}
        self.U = AdvancedPriorityQueue()

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
        for motion in _GRID8_MOTIONS:
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
        motion = get_motion_for_diff(diff.x, diff.y)
        if motion is None:
            return math.inf
        
        # incorporate optional occupancy cost
        cell_cost = self.cell_costs.get(to_node, 0.0)
        return edge_cost(cell_cost, motion.cost, alpha=self.alpha)

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
            self.U.add_or_update_task(node, self.calculate_key(node))

    def compute_shortest_path(self):
        """Main D* Lite computation loop."""
        iterations = 0
        max_iterations = self.width * self.height * 2  # More generous limit
        
        while (not self.U.empty() and
               (self.compare_keys(self.U.top_key(), self.calculate_key(self.start)) or
                self.get_rhs(self.start) != self.get_g(self.start))):

            iterations += 1
            if iterations > max_iterations:
                print(f"WARNING: compute_shortest_path exceeded {max_iterations} iterations")
                break

            node, k_old = self.U.pop_task()
            k_new = self.calculate_key(node)

            if self.compare_keys(k_old, k_new):
                # key changed — reinsert with new key
                self.U.add_or_update_task(node, k_new)
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
        
        if iterations > 0:
            print(f"compute_shortest_path completed in {iterations} iterations")
            print(f"Priority queue size: {self.U.size()}")
            print(f"Start {self.start} g-value: {self.get_g(self.start)}, rhs: {self.get_rhs(self.start)}")

    # ----------------------------- initialization / external API -----------------------------
    def initialize(self, start: GridNode, goal: GridNode):
        """Initialize the planner for first-time use."""
        self.start = start
        self.goal = goal
        self.last_start = start
        
        self.initialized = True
        self.U = AdvancedPriorityQueue()
        self.km = 0.0
        self.g_values.clear()
        self.rhs_values.clear()
        
        # Initialize goal
        self.set_rhs(self.goal, 0.0)
        self.U.add_or_update_task(self.goal, self.calculate_key(self.goal))

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
            print(f"Start moved, km updated to {self.km:.3f}")

    def set_goal(self, new_goal: GridNode):
        """Change goal and reinitialize."""
        if new_goal != self.goal:
            # Clear previous goal
            old_goal = self.goal
            self.set_rhs(old_goal, math.inf)
            
            # Set new goal
            self.goal = new_goal
            self.set_rhs(self.goal, 0.0)
            self.U.add_or_update_task(self.goal, self.calculate_key(self.goal))

    def force_recompute_region(self, center: GridNode, radius: int = 5):
        """Force recomputation of a region around a center point."""
        print(f"Force recomputing region around {center} with radius {radius}")
        
        nodes_updated = 0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                node = GridNode(center.x + dx, center.y + dy)
                if self.is_valid(node) and not self.is_obstacle(node):
                    # Reset g-values to force re-exploration
                    old_g = self.get_g(node)
                    if old_g < math.inf:
                        self.set_g(node, math.inf)
                        self.update_vertex(node)
                        nodes_updated += 1
        
        print(f"Reset {nodes_updated} nodes in region")
        self.compute_shortest_path()

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
        
        path = self.extract_path()
        
        # If path extraction failed, try forcing broader search
        if not path and self.get_g(self.start) < math.inf:
            print("Path extraction failed despite finite g-value - forcing broader search")
            self.force_recompute_region(self.start, radius=min(10, max(self.width, self.height) // 5))
            path = self.extract_path()
            
        return path

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
                
                # Skip successors with infinite g-values unless no alternatives
                if g_succ == math.inf:
                    continue
                    
                total_cost = edge_c + g_succ
                
                if total_cost < best_total_cost:
                    best_total_cost = total_cost
                    best_succ = succ
            
            # If no finite successors, trigger additional search
            if best_succ is None:
                print(f"Path extraction failed at {current} - need more search")
                print(f"Current g-value: {self.get_g(current)}, rhs: {self.get_rhs(current)}")
                
                # Check for unexplored successors
                has_unexplored = False
                for succ in self.get_successors(current):
                    if not self.is_obstacle(succ) and succ not in visited:
                        g_succ = self.get_g(succ)
                        edge_c = self.cost(current, succ)
                        if g_succ == math.inf and edge_c < math.inf:
                            has_unexplored = True
                            break
                
                if has_unexplored:
                    print("Triggering additional search to explore unexplored regions")
                    # Force broader search around current position
                    self.force_recompute_region(current, radius=8)
                    continue
                else:
                    print("No valid successors - path blocked")
                    return []
            
            current = best_succ

        if current == self.goal:
            path.append(self.goal)
            print(f"Successfully extracted path with {len(path)} nodes")
        else:
            print(f"Path extraction stopped after {iteration_count} iterations at {current}")
        
        return path

    def is_path_blocked(self, path: List[GridNode]) -> bool:
        """Check if any node in the path is now an obstacle."""
        return any(self.is_obstacle(node) for node in path)

    def update_dynamic_obstacles(self, added: Set[Tuple[int, int]], removed: Set[Tuple[int, int]]):
        """Update dynamic obstacles and trigger replanning with proper change propagation."""
        changed_nodes = set()
        actually_changed = False

        # Process added obstacles
        for o in added:
            if o not in self.dynamic_obstacles:
                self.dynamic_obstacles.add(o)
                n = GridNode(*o)
                if self.is_valid(n):
                    changed_nodes.add(n)
                    actually_changed = True
                    print(f"Added dynamic obstacle at {n}")

        # Process removed obstacles  
        for o in removed:
            if o in self.dynamic_obstacles:
                self.dynamic_obstacles.remove(o)
                n = GridNode(*o)
                if self.is_valid(n):
                    changed_nodes.add(n)
                    actually_changed = True
                    print(f"Removed dynamic obstacle at {n}")

        if not actually_changed:
            print("No actual changes to dynamic obstacles")
            return False

        print(f"Updating {len(changed_nodes)} changed nodes and their neighbors")
        
        # For each changed node, update all nodes that might be affected
        nodes_to_update = set()
        
        for node in changed_nodes:
            # Add the node itself
            nodes_to_update.add(node)
            
            # Add all neighbors - they might need to recompute paths through this node
            for nb in self.get_neighbors(node):
                if self.is_valid(nb):
                    nodes_to_update.add(nb)
            
            # For added obstacles: invalidate the node completely
            if (node.x, node.y) in added:
                if self.get_g(node) < math.inf:
                    print(f"Invalidating obstacle node {node} (was g={self.get_g(node):.2f})")
                    self.set_g(node, math.inf)
                    self.set_rhs(node, math.inf)
        
        # Update all affected nodes
        print(f"Total nodes to update: {len(nodes_to_update)}")
        for node in nodes_to_update:
            self.update_vertex(node)
        
        # Recompute shortest path
        print("Recomputing shortest path after dynamic obstacle update...")
        self.compute_shortest_path()
        return True


def visualize_grid(planner: DStarLite, path: List[GridNode] = None, figsize: Tuple[int, int] = (10, 10)):
    """Enhanced matplotlib visualization with better styling."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError:
        print("matplotlib not available - cannot visualize")
        return

    fig, ax = plt.subplots(figsize=figsize)
    
    # Create grid background
    grid = np.zeros((planner.height, planner.width, 3), dtype=np.float32)
    grid.fill(1.0)  # white background

    # Static obstacles (dark gray)
    for (x, y) in planner.obstacles:
        if 0 <= x < planner.width and 0 <= y < planner.height:
            grid[y, x] = [0.2, 0.2, 0.2]

    # Dynamic obstacles (red)
    for (x, y) in planner.dynamic_obstacles:
        if 0 <= x < planner.width and 0 <= y < planner.height:
            grid[y, x] = [0.8, 0.2, 0.2]

    ax.imshow(grid, origin='lower', extent=[-0.5, planner.width-0.5, -0.5, planner.height-0.5])

    # Draw path
    if path and len(path) > 1:
        xs = [n.x for n in path]
        ys = [n.y for n in path]
        ax.plot(xs, ys, 'b-', linewidth=3, alpha=0.8, label='Path')
        ax.scatter(xs[1:-1], ys[1:-1], c='blue', s=30, alpha=0.6, zorder=4)

    # Start and goal
    ax.scatter([planner.start.x], [planner.start.y], c='green', s=200, marker='*', 
              label='Start', zorder=5, edgecolors='darkgreen', linewidth=2)
    ax.scatter([planner.goal.x], [planner.goal.y], c='red', s=200, marker='X', 
              label='Goal', zorder=5, edgecolors='darkred', linewidth=2)
    
    ax.set_xlim(-0.5, planner.width - 0.5)
    ax.set_ylim(-0.5, planner.height - 0.5)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.legend(loc='upper right')
    ax.set_title(f'D* Lite Grid Planner (Path length: {len(path) if path else 0})')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def demo_comprehensive():
    """Comprehensive demo showcasing various D* Lite capabilities."""
    print("Running comprehensive D* Lite demo...")
    
    # Create a more interesting environment
    width, height = 60, 40
    
    # Create maze-like obstacles
    obstacles = set()
    
    # Outer walls
    for x in range(width):
        obstacles.add((x, 0))
        obstacles.add((x, height-1))
    for y in range(height):
        obstacles.add((0, y))
        obstacles.add((width-1, y))
    
    # Internal walls creating passages
    for y in range(5, 25):
        obstacles.add((15, y))
    for y in range(15, 35):
        obstacles.add((30, y))
    for x in range(20, 40):
        obstacles.add((x, 20))
    
    planner = DStarLite(width, height, obstacles)
    
    start = GridNode(5, 5)
    goal = GridNode(50, 30)

    # Test 1: Initial planning
    print("\n=== Test 1: Initial Planning ===")
    path = planner.plan_path(start, goal)
    if path:
        print(f"Initial path found with {len(path)} nodes")
        print(f"Path cost estimate: {len(path):.1f}")
        visualize_grid(planner, path)
    else:
        print("No initial path found - check obstacle configuration")
        return

    # Test 2: Dynamic obstacle that blocks path
    print("\n=== Test 2: Dynamic Obstacle Blocking Path ===")
    if len(path) > 10:
        # Block a section of the path
        mid_section = path[len(path)//3:len(path)//3+5]
        block_obstacles = {(n.x, n.y) for n in mid_section[:8]}
        block_obstacles = {(20,5+i) for i in range(5)}  # Block a vertical section
        block_obstacles.update({(30,10+i) for i in range(7)})
        print(f"Adding obstacles at: {block_obstacles}")
        
        planner.update_dynamic_obstacles(block_obstacles, set())
        
        new_path = planner.extract_path()
        if new_path:
            print(f"Replanned path found with {len(new_path)} nodes")
            visualize_grid(planner, new_path)
        else:
            print("No path after blocking - trying force recompute")
            planner.force_recompute_region(planner.start, radius=15)
            new_path = planner.extract_path()
            if new_path:
                print(f"Path found after force recompute: {len(new_path)} nodes")
                visualize_grid(planner, new_path)

    # Test 3: Moving start position
    print("\n=== Test 3: Moving Start Position ===")
    if path and len(path) > 5:
        new_start = path[3]  # Move partway along the path
        print(f"Moving start from {planner.start} to {new_start}")
        
        updated_path = planner.plan_path(new_start, goal)
        if updated_path:
            print(f"Updated path from new start: {len(updated_path)} nodes")
            visualize_grid(planner, updated_path)

    # Test 4: Removing obstacles to create shortcuts
    print("\n=== Test 4: Removing Obstacles (Creating Shortcuts) ===")
    # Remove some wall sections to create shortcuts
    shortcut_removals = {(15, 12), (15, 13), (15, 14)}
    print(f"Removing obstacles at: {shortcut_removals}")
    
    planner.update_dynamic_obstacles(set(), shortcut_removals | block_obstacles)
    
    final_path = planner.extract_path()
    if final_path:
        print(f"Final optimized path: {len(final_path)} nodes")
        visualize_grid(planner, final_path)
    
    print("\n=== Demo Complete ===")
    print(f"Final statistics:")
    print(f"- Grid size: {width} x {height}")
    print(f"- Static obstacles: {len(obstacles)}")
    print(f"- Dynamic obstacles: {len(planner.dynamic_obstacles)}")
    print(f"- G-values computed: {len(planner.g_values)}")
    print(f"- RHS values computed: {len(planner.rhs_values)}")


if __name__ == '__main__':
    demo_comprehensive()