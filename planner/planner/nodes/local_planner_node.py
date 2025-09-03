import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
import numpy as np
import matplotlib.pyplot as plt

from planner.algorithms.hybrid_a_star import hybrid_a_star_planning 
from utils.car import move, check_car_collision, MAX_STEER, WB, plot_car, BUBBLE_R
import utils.reeds_shepp_path_planning as rs

# Local planner parameters
XY_GRID_RESOLUTION = 1  # meters per grid cell
YAW_GRID_RESOLUTION = np.deg2rad(15)  # radians per yaw step
SHOW_ANIMATION = False
DOWNSAMPLE_OBSTACLES_TO = 200  # max number of obstacles to keep for local planning
Local_Grid_Width= 20.0 # meters : local planner window size
Local_Grid_Height= 20.0 # meters: local planner window size
iter = 0  # global iteration counter for logging

class LocalHybridAStarNode(Node):
    def __init__(self):
        super().__init__('local_hybrid_astar_node')

        # Subscribers
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10
        )
        self.pose_sub = self.create_subscription(
            PoseStamped, '/odom', self.pose_callback, 10
        )
        self.global_path = None
        self.global_path_sub = self.create_subscription(
            Path,
            '/global_path',   # replace with your global path topic
            self.global_path_callback,
            10)

        self.iter = 0  # global iteration counter for logging

        # Publisher
        self.path_pub = self.create_publisher(Path, '/local_path', 10)
        # Vehicle state
        self.current_pose = [1, 1, np.deg2rad(0.0)]
        self.goal = [10, 13, np.deg2rad(0.0)]

        # Map storage
        self.map_msg = None
        self.ox, self.oy = [], []

        # Plotting
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.im = None
        self.path_line, = self.ax.plot([], [], 'b.-', label="Path")
        self.start_scatter = self.ax.plot([], [], "go", markersize=10, label="Start")[0]
        self.goal_scatter = self.ax.plot([], [], "ro", markersize=10, label="Goal")[0]

        self.ax.legend()
        plt.ion()   # interactive mode
        plt.show()

    
    def pose_callback(self, msg: PoseStamped):
        self.current_pose = [msg.pose.position.x,
                             msg.pose.position.y,
                             np.arctan2(2*(msg.pose.orientation.w*msg.pose.orientation.z),
                                        1-2*msg.pose.orientation.z**2)]
        
    def map_callback(self, msg: OccupancyGrid):
        self.map_msg = msg
        res = msg.info.resolution

        # convert occupancy grid to 2D numpy array
        grid = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        ys, xs = np.nonzero(grid > 50)

        # convert to world coords (cell centers)
        ox_full = xs * res + msg.info.origin.position.x + res/2.0
        oy_full = ys * res + msg.info.origin.position.y + res/2.0



        # if we don't have pose yet, just store full lists (or return)
        if self.current_pose is None:
            self.ox = ox_full.tolist()
            self.oy = oy_full.tolist()
            return

        cx, cy = self.current_pose[0], self.current_pose[1]
        half_w, half_h = Local_Grid_Width / 2.0, Local_Grid_Height / 2.0

        # mask obstacles inside the local window
        mask = np.logical_and.reduce((
            ox_full >= cx - half_w,
            ox_full <= cx + half_w,
            oy_full >= cy - half_h,
            oy_full <= cy + half_h
        ))
        ox_local = ox_full[mask]
        oy_local = oy_full[mask]
        min_x = cx - half_w   # leftmost boundary of local window
        max_x = cx + half_w   # rightmost boundary
        min_y = cy - half_h   # bottom boundary
        max_y = cy + half_h   # top boundary

        # downsample if there are too many points
        n_pts = len(ox_local)
        if n_pts > DOWNSAMPLE_OBSTACLES_TO:
            idx = np.random.choice(n_pts, DOWNSAMPLE_OBSTACLES_TO, replace=False)
            ox_local = ox_local[idx]
            oy_local = oy_local[idx]

        # ensure lists are non-empty (Config expects min/max)
        if ox_local.size == 0:
            # add four window corners so Config has bounds
            ox_local = np.array([cx - half_w, cx - half_w, cx + half_w, cx + half_w])
            oy_local = np.array([cy - half_h, cy + half_h, cy - half_h, cy + half_h])

        self.ox = ox_local.tolist()
        self.oy = oy_local.tolist()

        # choose local goal (furthest global path point inside window)
        local_goal = self.get_local_goal()
        if local_goal is None:
            self.get_logger().warn("No local goal (global path empty or none in window)")
            return

        start = [
        self.current_pose[0] * self.map_msg.info.resolution + self.map_msg.info.origin.position.x/2,
        self.current_pose[1] * self.map_msg.info.resolution + self.map_msg.info.origin.position.y/2,
        self.current_pose[2]  
        ]
        goal = local_goal

        if not (min_x <= start[0] <= max_x and min_y <= start[1] <= max_y):
            self.get_logger().warn("Start outside local window, skipping planning")
            return
        else:
            print(f"Local planner start inside window: {start}")
        if not (min_x <= goal[0] <= max_x and min_y <= goal[1] <= max_y):
            self.get_logger().warn("Goal outside local window, skipping planning")
            return
        else:
            print(f"Local planner goal inside window: {goal}")
        

        # debug prints
        self.get_logger().info(f"Local planner: start={start}, goal={goal}, obstacles={len(self.ox)}")
        
        self.plan_path(start, goal)

    
    def plan_path(self,start, goal):
        # run planner
        path_result = hybrid_a_star_planning(start, goal, self.ox, self.oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)

        if path_result and len(path_result.x_list) > 0:
            self.publish_path(path_result)
            if SHOW_ANIMATION:
                self.plot_path(path_result)
    
    def publish_path(self, path_result):
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        for x, y, yaw in zip(path_result.x_list, path_result.y_list, path_result.yaw_list):
            pose = PoseStamped()
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            # yaw -> quaternion
            pose.pose.orientation.w = np.cos(yaw / 2)
            pose.pose.orientation.z = np.sin(yaw / 2)
            path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)
    
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
            self.start_scatter.set_data([self.current_pose[0]], [self.current_pose[1]])
        else:
            self.start_scatter.set_data([], [])

        # update goal
        if goal:
            self.goal_scatter.set_data([self.goal[0]], [self.goal[1]])
        else:
            self.goal_scatter.set_data([], [])

        self.ax.set_title("ROS2 Occupancy Grid with D* Lite Path")
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def plot_path(self, path_result):       
        plt.plot(self.ox, self.oy, ".k")
        rs.plot_arrow(self.current_pose[0], self.current_pose[1], self.current_pose[2], fc='g')
        rs.plot_arrow( self.goal[0],  self.goal[1],  self.goal[2])

        plt.grid(True)
        plt.axis("equal")

        x = path_result.x_list
        y = path_result.y_list
        yaw = path_result.yaw_list

        for i_x, i_y, i_yaw in zip(x, y, yaw):
            plt.cla()
            plt.plot(self.ox, self.oy, ".k")
            plt.plot(x, y, "-r", label="Hybrid A* path")
            plt.grid(True)
            plt.axis("equal")
            plot_car(i_x, i_y, i_yaw)
            plt.pause(0.0001)


    def get_local_goal(self):
        """
        Take the furthest global path point within the local window
        or fallback to the last global point.
        """
        if not self.global_path or len(self.global_path.poses) == 0:
            self.get_logger().warn("No global path available")
            return None

        x_c, y_c = self.current_pose[0], self.current_pose[1]
        half_w, half_h = Local_Grid_Width / 2, Local_Grid_Height / 2

        # Iterate reversed to get furthest point first
        for pose in reversed(self.global_path.poses):
            gx = pose.pose.position.x
            gy = pose.pose.position.y
            # Check if inside local window
            if (x_c - half_w <= gx <= x_c + half_w) and (y_c - half_h <= gy <= y_c + half_h):
                yaw = self.current_pose[2]  # optional: could compute heading from path
                self.get_logger().info(f"Local goal from global path: iter{self.iter}({gx}, {gy})")
                self.iter += 1
                return [gx, gy, yaw]

        # Fallback: last global point
        last_pose = self.global_path.poses[-1]
        print(f"Fallback to last global goal: ({last_pose.pose.position.x}, {last_pose.pose.position.y})")
        return [last_pose.pose.position.x, last_pose.pose.position.y, self.current_pose[2]]
    
    def global_path_callback(self, msg: Path):
        """
        Store the latest global path received from the global planner.
        """
        self.global_path = msg
        print(f"gloal_path_callback triggered {len(msg.poses)}")
        self.get_logger().info(f"Received global path with {len(msg.poses)} points")
def main(args=None):
    rclpy.init(args=args)
    node = LocalHybridAStarNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
