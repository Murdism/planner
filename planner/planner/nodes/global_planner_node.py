import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
import numpy as np
import matplotlib.pyplot as plt


from planner.algorithms.dstar_lite import DStarLite, GridNode  # adjust import to your file structure


START = GridNode(1, 1)  # set your start here
GOAL = GridNode(10, 13)  # set your goal here

class DStarLiteNode(Node):
    def __init__(self, start: GridNode, goal: GridNode):
        super().__init__('dstar_lite_node')

        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10
        )
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/odom',
            self.pose_callback,
            10
        )

        self.path_nodes = None
        self.viz = False

        # Publishers
        self.path_pub = self.create_publisher(Path, '/global_path', 10)

        # D* Lite setup
        self.dstar = None  # will initialize after first map received
        self.current_start = start
        self.goal = goal

        self.map_msg = None

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.im = None
        self.path_line, = self.ax.plot([], [], 'b.-', label="Path")
        self.start_scatter = self.ax.plot([], [], "go", markersize=10, label="Start")[0]
        self.goal_scatter = self.ax.plot([], [], "ro", markersize=10, label="Goal")[0]

        self.ax.legend()
        plt.ion()   # interactive mode
        plt.show()

    def pose_callback(self, msg: PoseStamped):
        """Update vehicle start position in grid coordinates."""
        if not self.map_msg:
            return

        info = self.map_msg.info
        res = info.resolution
        ox0, oy0 = info.origin.position.x, info.origin.position.y

        x_idx = int((msg.pose.position.x - ox0) / res)
        y_idx = int((msg.pose.position.y - oy0) / res)

        new_start = GridNode(x_idx, y_idx)

        if self.dstar:
            self.dstar.set_start(new_start)
        self.current_start = new_start

    def map_callback(self, msg: OccupancyGrid):
        """Handle incoming occupancy grid map efficiently."""

        print("Map callback triggered")

        self.map_msg = msg  # store latest map

        info = msg.info
        res = info.resolution
        ox0, oy0 = info.origin.position.x, info.origin.position.y
        h, w = info.height, info.width

        # Convert occupancy grid to 2D numpy array
        grid = np.array(msg.data, dtype=np.int8).reshape(h, w)
        ys, xs = np.nonzero(grid >50)
        new_obstacles = set(zip(xs, ys))
        self.get_logger().info(f"Map received: {w}x{h}, Obstacles: {len(new_obstacles)}")
        start = self.current_start or GridNode(0, 0)
        # self.visualize_ros_grid(grid, [], start=start, goal=self.goal)
        # Initialize D* Lite on first map
        if not self.dstar:
            start = self.current_start or GridNode(1, 1)
            self.dstar = DStarLite(w, h, obstacles=new_obstacles)
            self.dstar.initialize(start, self.goal)
            self.path_nodes = self.dstar.extract_path()
            if self.viz:
                self.visualize_ros_grid(grid, self.path_nodes, start=start, goal=self.goal)
            self.publish_path(self.path_nodes, ox0, oy0, res)
            return

        added = new_obstacles - self.dstar.dynamic_obstacles
        removed = self.dstar.dynamic_obstacles - new_obstacles
        print(f"Dynamic obstacles changed. Added: {len(added)}, Removed: {len(removed)}")

        if added or removed:
            self.dstar.update_dynamic_obstacles(added, removed)
            self.path_nodes = self.dstar.extract_path()
            if self.viz:
                self.visualize_ros_grid(grid, self.path_nodes, start=self.current_start, goal=self.goal)
        self.publish_path(self.path_nodes, ox0, oy0, res)

    def visualize_ros_grid(self, grid, path=None, start=None, goal=None):
        """Update the visualization without creating new figures."""

        # update grid
        if self.im is None:
            self.im = self.ax.imshow(grid, cmap="gray_r", origin="lower")
        else:
            self.im.set_data(grid)

        # update path
        if path:
            xs = [n.x for n in path]
            ys = [n.y for n in path]
            self.path_line.set_data(xs, ys)
        else:
            self.path_line.set_data([], [])

        # update start
        if start:
            self.start_scatter.set_data([start.x], [start.y])
        else:
            self.start_scatter.set_data([], [])

        # update goal
        if goal:
            self.goal_scatter.set_data([goal.x], [goal.y])
        else:
            self.goal_scatter.set_data([], [])

        self.ax.set_title("ROS2 Occupancy Grid with D* Lite Path")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def publish_path(self, path_nodes, ox0, oy0, res):
        path = Path()
        path.header.frame_id = "map"
        for node in path_nodes:
            pose = PoseStamped()
            pose.pose.position.x = node.x * res + ox0
            pose.pose.position.y = node.y * res + oy0
            pose.pose.position.z = 0.0
            path.poses.append(pose)
        self.path_pub.publish(path)
        self.get_logger().info(f"Published path with {len(path.poses)} points")
def main(args=None):

    rclpy.init(args=args)
    node = DStarLiteNode(START, GOAL)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()











