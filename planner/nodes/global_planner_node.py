import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
import numpy as np
import matplotlib.pyplot as plt

from planner.algorithms.dstar_lite import DStarLite, GridNode  # adjust to your layout


class DStarLiteNode(Node):
    def __init__(self):
        super().__init__('dstar_lite_node')

        # --- State ---
        self.map_msg = None
        self.grid = None
        self.obstacles = set()

        # raw poses from topics (as-is)
        self.start_raw = None  # (x,y) from /robot
        self.goal_raw  = None  # (x,y) from /goal

        # flipped nodes used by planner (after applying (W-1-x, H-1-y))
        self.start_node = None  # GridNode
        self.goal_node  = None  # GridNode

        self.dstar = None
        self.path_nodes = None

        # --- Subscribers ---
        self.create_subscription(OccupancyGrid, '/map',   self.map_callback,   10)
        self.create_subscription(PoseStamped,   '/robot', self.robot_callback, 10)
        self.create_subscription(PoseStamped,   '/goal',  self.goal_callback,  10)

        # --- Publisher ---
        self.path_pub = self.create_publisher(Path, '/global_path', 10)

        # create a timer to periodically publish the plan
        self.timer = self.create_timer(0.1, self.publish_path)

        # --- Viz ---
        self.viz = True
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.im = None
        self.path_line, = self.ax.plot([], [], 'b.-', label="Path")
        self.start_scatter = self.ax.plot([], [], "go", markersize=10, label="Start")[0]
        self.goal_scatter  = self.ax.plot([], [], "ro", markersize=10, label="Goal")[0]
        self.ax.legend()
        plt.ion()
        plt.show()

    # -------------------- Helpers --------------------

    def _flip_xy(self, x: float, y: float):
        """Apply same flips as mapping node: I'=(W-1)-I, J'=(H-1)-J."""
        if self.map_msg is None:
            return None
        w = self.map_msg.info.width
        h = self.map_msg.info.height
        return (x, (h - 1) - y )

    def _build_obstacles(self, msg: OccupancyGrid):
        """Grid>50 considered occupied; returns ndarray and set of (x,y) indices."""
        h, w = msg.info.height, msg.info.width
        grid = np.array(msg.data, dtype=np.int8).reshape(h, w)
        ys, xs = np.nonzero(grid > 50)
        return grid, set(zip(xs, ys))

    def _update_flipped_nodes(self):
        """Recompute planner start/goal (flipped) from raw when map dims are known."""
        changed = False
        if self.map_msg is None:
            return changed

        if self.start_raw is not None:
            fx, fy = self._flip_xy(*self.start_raw)
            node = GridNode(fx, fy)
            if self.start_node != node:
                self.start_node = node
                changed = True

        if self.goal_raw is not None:
            fx, fy = self._flip_xy(*self.goal_raw)
            node = GridNode(fx, fy)
            if self.goal_node != node:
                self.goal_node = node
                changed = True

        return changed

    def _ensure_planner(self):
        """Initialize D* Lite once we have map + flipped start + flipped goal."""
        if self.dstar or self.map_msg is None or self.start_node is None or self.goal_node is None:
            return False

        self.grid, self.obstacles = self._build_obstacles(self.map_msg)
        w = self.map_msg.info.width
        h = self.map_msg.info.height

        self.dstar = DStarLite(w, h, obstacles=self.obstacles)
        self.dstar.initialize(self.start_node, self.goal_node)

        self.path_nodes = self.dstar.extract_path()
        if self.viz:
            # For visualization we draw the flipped nodes (the ones the planner uses)
            self.visualize_ros_grid(self.grid, self.path_nodes,
                                    start=self.start_node, goal=self.goal_node)
        # self.publish_path(self.path_nodes)  # as-is
        self.get_logger().info("D* Lite initialized (with start/goal flips).")
        return True

    def _replan(self, reason: str):
        """Re-plan using current flipped start/goal and latest obstacles."""
        if self.map_msg is None or self.start_node is None or self.goal_node is None:
            return

        self.grid, new_obstacles = self._build_obstacles(self.map_msg)
        print(f"start from global {self.start_node.x,self.start_node.y} goal: {self.goal_node.x} {self.goal_node.y}")

        if self.dstar is None:
            self._ensure_planner()
            return

        # Update goal/start if API supports; else re-init
        if hasattr(self.dstar, "set_goal"):
            self.dstar.set_goal(self.goal_node)
        else:
            self.dstar = DStarLite(self.map_msg.info.width, self.map_msg.info.height, obstacles=new_obstacles)
            self.dstar.initialize(self.start_node, self.goal_node)

        if hasattr(self.dstar, "set_start"):
            self.dstar.set_start(self.start_node)

        # Dynamic obstacles
        if hasattr(self.dstar, "update_dynamic_obstacles"):
            added   = new_obstacles - self.obstacles
            removed = self.obstacles - new_obstacles
            if added or removed:
                self.dstar.update_dynamic_obstacles(added, removed)
        self.obstacles = new_obstacles

        # Extract path & publish
        self.path_nodes = self.dstar.extract_path()
        if self.viz:
            self.visualize_ros_grid(self.grid, self.path_nodes,
                                    start=self.start_node, goal=self.goal_node)
        # self.publish_path(self.path_nodes)
        self.get_logger().info(f"Replanned ({reason}); points: {len(self.path_nodes) if self.path_nodes else 0}")

    # -------------------- Callbacks --------------------

    def robot_callback(self, msg: PoseStamped):
        """Take /robot as-is, then apply same flips as mapping grid."""
        self.start_raw = (msg.pose.position.x, msg.pose.position.y)
        if self._update_flipped_nodes():
            if self._ensure_planner():
                return
            if self.dstar:
                self._replan("start moved")

    def goal_callback(self, msg: PoseStamped):
        """Take /goal as-is, then apply same flips as mapping grid."""
        self.goal_raw = (msg.pose.position.x, msg.pose.position.y)
        if self._update_flipped_nodes():
            if self._ensure_planner():
                return
            if self.dstar:
                self._replan("goal changed")

    def map_callback(self, msg: OccupancyGrid):
        """On new map: rebuild obstacles; if dims changed, recompute flips and replan."""
        self.map_msg = msg
        # dims_changed = self._update_flipped_nodes()
        # if self._ensure_planner():
        #     return

        # if self.dstar:
        #     self.grid, new_obstacles = self._build_obstacles(msg)
        #     added   = new_obstacles - self.obstacles
        #     removed = self.obstacles - new_obstacles
        #     self.obstacles = new_obstacles

        #     need_replan = bool(added or removed or dims_changed)
        #     if need_replan:
        #         if hasattr(self.dstar, "update_dynamic_obstacles") and (added or removed):
        #             self.dstar.update_dynamic_obstacles(added, removed)
        #         self.path_nodes = self.dstar.extract_path()
        #         if self.viz:
        #             self.visualize_ros_grid(self.grid, self.path_nodes,
        #                                     start=self.start_node, goal=self.goal_node)
        #         self.publish_path(self.path_nodes)

    # -------------------- Viz & Path Publishing (as-is) --------------------

    def visualize_ros_grid(self, grid, path=None, start=None, goal=None):
        """Update the visualization (grid origin lower to match y-up)."""
        if self.im is None:
            self.im = self.ax.imshow(grid, cmap="gray_r", origin="lower")
        else:
            self.im.set_data(grid)

        if path:
            xs = [n.x for n in path]
            ys = [n.y for n in path]
            self.path_line.set_data(xs, ys)
        else:
            self.path_line.set_data([], [])

        if start:
            self.start_scatter.set_data([start.x], [start.y])
        else:
            self.start_scatter.set_data([], [])

        if goal:
            self.goal_scatter.set_data([goal.x], [goal.y])
        else:
            self.goal_scatter.set_data([], [])

        self.ax.set_title("ROS2 Occupancy Grid with D* Lite Path (flipped start/goal)")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def publish_path(self):
        """Publish Path with poses directly from node.x/node.y (no transforms)."""

        if self.path_nodes is None:
            self.get_logger().info("No path to publish.")
            return
        path = Path()
        path.header.frame_id = "map"
        path.header.stamp = self.get_clock().now().to_msg()
        
        if self.path_nodes:
            for n in self.path_nodes:
                # revert flip to map coordinates
                h = self.map_msg.info.height
                xm = n.x
                ym = (h - 1) - n.y

                ps = PoseStamped()
                ps.header = path.header
                ps.pose.position.x = float(xm)
                ps.pose.position.y = float(ym)
                ps.pose.position.z = 0.0
                path.poses.append(ps)
        self.path_pub.publish(path)
        self.get_logger().info(f"Published path with {len(path.poses)} points (as-is).")
    def _grid_to_world(self, ix: int, iy_planner: int):
        """Grid indices (ix, iy_planner) -> map/world meters (cell centers)."""
        assert self.map_msg is not None
        info = self.map_msg.info
        res = info.resolution
        ox0 = info.origin.position.x
        oy0 = info.origin.position.y
        h   = info.height

        iy_nom = (h - 1) - iy_planner               # unflip
        x_w = ix * res + ox0 + 0.5 * res
        y_w = iy_nom * res + oy0 + 0.5 * res
        return float(x_w), float(y_w)

def main(args=None):
    rclpy.init(args=args)
    node = DStarLiteNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
