import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose
import numpy as np
import random

class MapPublisher(Node):
    def __init__(self):
        super().__init__('map_publisher')

        self.publisher_ = self.create_publisher(OccupancyGrid, '/map', 10)
        self.goal = (10,13) # goal can't be set as obstacle

        # Timers
        self.publish_timer = self.create_timer(0.5, self.publish_map)   # publish every 1s
        self.update_timer = self.create_timer(5.0, self.update_obstacles)  # change map every 5s

        # Map parameters
        self.resolution = 1.0
        self.width = 60
        self.height = 120
        self.origin_x = 0.0
        self.origin_y = 0.0

        # Start with empty map
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)

        # Keep track of a toggle index
        self.step = 0

    def update_obstacles(self):
        """Change obstacles every 5 seconds (like demo)."""
        self.grid.fill(0)  # reset to free

        if self.step % 3 == 0:
            # Case 1: block in center
            self.grid[5:10, 5:10] = 100
            self.grid[12, :-8] = 100
        elif self.step % 3 == 1:
            # Case 2: horizontal wall
            self.grid[12, :-2] = 100
        else:
            # Case 3: random scattered obstacles
            
            # for _ in range(4):
            #     x = random.randint(0, self.width - 1)
            #     y = random.randint(0, self.height - 1)
            #     self.grid[y, x] = 100
            
            self.grid[12, :-8] = 100
        self.grid[self.goal[1], self.goal[0]] = 0  # ensure goal is free
        self.get_logger().info(f"Updated obstacles (pattern {self.step % 3})")
        self.step += 1

    def publish_map(self):
        msg = OccupancyGrid()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.info = MapMetaData()
        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height

        origin = Pose()
        origin.position.x = self.origin_x
        origin.position.y = self.origin_y
        origin.position.z = 0.0
        msg.info.origin = origin

        msg.data = self.grid.flatten().tolist()

        self.publisher_.publish(msg)
        self.get_logger().info("Published occupancy grid")

def main(args=None):
    rclpy.init(args=args)
    node = MapPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
