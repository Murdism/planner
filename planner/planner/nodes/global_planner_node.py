import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
import numpy as np

from core.dstar_lite import DStarLite, GridNode  # adjust import to your file structure


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

        # Publishers
        self.path_pub = self.create_publisher(Path, '/dstar_path', 10)

        # D* Lite setup
        self.goal = goal
        self.dstar = None  # will initialize after first map received
        self.current_start = start

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
    goal = GridNode(10, 10)  # set your goal here
    start = GridNode(0, 0)  # set your start here
    rclpy.init(args=args)
    node = DStarLiteNode(start, goal)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
















# '''
# Ros2 based global planner node

# This node implements a global planner using D* Lite algorithm.'''

# import rclpy
# from rclpy.node import Node
# from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoS
# from nav_msgs.msg import OccupancyGrid, Path
# from geometry_msgs.msg import PoseStamped
# from std_msgs.msg import Header
# import numpy as np
# import math

# from planner.core.d_star_lite import DStarLite

# class GlobalPlannerNode(Node):
#     def __init__(self,start=(0,0),goal=(0,0),ros_based=True):
#         super().__init__('global_planner_node')
#         self.get_logger().info('Global Planner Node has been started.')
#         self.get_logger().info('Using D* Lite with start: {} and goal: {}'.format(start, goal))

#         # Initialize D* Lite planner
#         self.planner = DStarLite()

#         # Initialize variables
#         self.map_data = None
#         self.start = start
#         self.goal = goal

#         # Setup subscribers and publishers
#         if ros_based:
#             self.setup_subscribers()
#         self.setup_publishers()

#     def setup_subscribers(self):
#         qos_profile = QoSProfile(
#             reliability=QoSReliabilityPolicy.BEST_EFFORT,
#             depth=10
#         )
#         self.map_subscriber = self.create_subscription(
#             OccupancyGrid,
#             '/map',
#             self.map_callback,
#             qos_profile
#         )
#         self.start_subscriber = self.create_subscription(
#             PoseStamped,
#             '/start',
#             self.start_callback,
#             qos_profile
#         )
#         self.goal_subscriber = self.create_subscription(
#             PoseStamped,
#             '/goal',
#             self.goal_callback,
#             qos_profile
#         )

#     def setup_publishers(self):
#         self.path_publisher = self.create_publisher(
#             Path,
#             '/planned_path',
#             10
#         )
#     def map_callback(self, msg):
#         self.get_logger().info('Received new map')
#         width = msg.info.width
#         height = msg.info.height
#         data = np.array(msg.data).reshape((height, width))
#         self.map_data = data
#         self.planner.set_map(data)

#     def start_callback(self, msg):
#         self.get_logger().info('Received new start position')
#         x = int(msg.pose.position.x)
#         y = int(msg.pose.position.y)
#         self.start = (x, y)
#         self.planner.set_start(self.start)
#         self.try_plan_path()

#     def goal_callback(self, msg):
#         self.get_logger().info('Received new goal position')
#         x = int(msg.pose.position.x)
#         y = int(msg.pose.position.y)
#         self.goal = (x, y)
#         self.planner.set_goal(self.goal)
#         self.try_plan_path()

#     def try_plan_path(self):
#         if self.map_data is not None and self.start is not None and self.goal is not None:
#             path = self.planner.plan()
#             if path:
#                 self.publish_path(path)
#             else:
#                 self.get_logger().warn('No path found')

#     def publish_path(self, path):
#         path_msg = Path()
#         path_msg.header = Header()
#         path_msg.header.stamp = self.get_clock().now().to_msg()
#         path_msg.header.frame_id = 'map'

#         for (x, y) in path:
#             pose = PoseStamped()
#             pose.header = path_msg.header
#             pose.pose.position.x = x
#             pose.pose.position.y = y
#             pose.pose.position.z = 0.0
#             pose.pose.orientation.w = 1.0
#             path_msg.poses.append(pose)         
#         self.path_publisher.publish(path_msg)
#         self.get_logger().info('Published planned path')    



# def main(args=None):
#     start = (0, 0)  # Example start position
#     goal = (10, 10)  # Example goal position
#     ros_based = True  # Reads occupancy grid and location from ROS topics

#     rclpy.init(args=args)
#     global_planner_node = GlobalPlannerNode(start, goal, ros_based)
#     rclpy.spin(global_planner_node)
#     global_planner_node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()
