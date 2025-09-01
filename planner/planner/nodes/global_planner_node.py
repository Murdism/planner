import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
import numpy as np

from core.dstar_lite import DStarLite, GridNode  # adjust import to your file structure


class DStarLiteNode(Node):
    def __init__(self,start=GridNode(0,0),goal=GridNode(0,0)):
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
            '/vehicle_pose',
            self.pose_callback,
            10
        )

        # Publishers
        self.path_pub = self.create_publisher(Path, '/dstar_path', 10)

        # D* Lite setup
        self.goal = goal  # set your goal here
        self.dstar = None  # will initialize after first map received
        self.current_start = start  # will update with vehicle pose

        self.map_msg = None

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
        self.map_msg = msg  # store latest map

        info = msg.info
        res = info.resolution
        ox0, oy0 = info.origin.position.x, info.origin.position.y
        h, w = info.height, info.width

        # Convert occupancy grid to 2D numpy array
        grid = np.array(msg.data, dtype=np.int8).reshape(h, w)
        ys, xs = np.nonzero(grid > 50)
        new_obstacles = set(zip(xs, ys))

        # Initialize D* Lite on first map
        if not self.dstar:
            start = self.current_start or GridNode(0, 0)
            self.dstar = DStarLite(w, h, obstacles=new_obstacles)
            self.dstar.initialize(start, self.goal)
            path_nodes = self.dstar.extract_path()
            self.publish_path(path_nodes, ox0, oy0, res)
            return

        added = new_obstacles - self.dstar.dynamic_obstacles
        removed = self.dstar.dynamic_obstacles - new_obstacles

        if added or removed:
            self.dstar.update_dynamic_obstacles(added, removed)
            path_nodes = self.dstar.extract_path()
            self.publish_path(path_nodes, ox0, oy0, res)

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


def main(args=None):
    
    start=GridNode(0,0)
    goal=GridNode(10,10)

    rclpy.init(args=args)
    node = DStarLiteNode(start,goal)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':

    main()
