'''
Ros2 based global planner node

This node implements a global planner using D* Lite algorithm.'''

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoS
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
import numpy as np
import math

from planner.core.d_star_lite import DStarLite

class GlobalPlannerNode(Node):
    def __init__(self,start=(0,0),goal=(0,0),ros_based=True):
        super().__init__('global_planner_node')
        self.get_logger().info('Global Planner Node has been started.')
        self.get_logger().info('Using D* Lite with start: {} and goal: {}'.format(start, goal))

        # Initialize D* Lite planner
        self.planner = DStarLite()

        # Initialize variables
        self.map_data = None
        self.start = start
        self.goal = goal

        # Setup subscribers and publishers
        if ros_based:
            self.setup_subscribers()
        self.setup_publishers()

    def setup_subscribers(self):
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            depth=10
        )
        self.map_subscriber = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            qos_profile
        )
        self.start_subscriber = self.create_subscription(
            PoseStamped,
            '/start',
            self.start_callback,
            qos_profile
        )
        self.goal_subscriber = self.create_subscription(
            PoseStamped,
            '/goal',
            self.goal_callback,
            qos_profile
        )

    def setup_publishers(self):
        self.path_publisher = self.create_publisher(
            Path,
            '/planned_path',
            10
        )
    def map_callback(self, msg):
        self.get_logger().info('Received new map')
        width = msg.info.width
        height = msg.info.height
        data = np.array(msg.data).reshape((height, width))
        self.map_data = data
        self.planner.set_map(data)

    def start_callback(self, msg):
        self.get_logger().info('Received new start position')
        x = int(msg.pose.position.x)
        y = int(msg.pose.position.y)
        self.start = (x, y)
        self.planner.set_start(self.start)
        self.try_plan_path()

    def goal_callback(self, msg):
        self.get_logger().info('Received new goal position')
        x = int(msg.pose.position.x)
        y = int(msg.pose.position.y)
        self.goal = (x, y)
        self.planner.set_goal(self.goal)
        self.try_plan_path()

    def try_plan_path(self):
        if self.map_data is not None and self.start is not None and self.goal is not None:
            path = self.planner.plan()
            if path:
                self.publish_path(path)
            else:
                self.get_logger().warn('No path found')

    def publish_path(self, path):
        path_msg = Path()
        path_msg.header = Header()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = 'map'

        for (x, y) in path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)         
        self.path_publisher.publish(path_msg)
        self.get_logger().info('Published planned path')    



def main(args=None):
    start = (0, 0)  # Example start position
    goal = (10, 10)  # Example goal position
    ros_based = True  # Reads occupancy grid and location from ROS topics

    rclpy.init(args=args)
    global_planner_node = GlobalPlannerNode(start, goal, ros_based)
    rclpy.spin(global_planner_node)
    global_planner_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()