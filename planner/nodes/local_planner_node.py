import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped
import numpy as np
import matplotlib.pyplot as plt

from planner.algorithms.hybrid_a_star import hybrid_a_star_planning 
from planner.utils.car import move, check_car_collision, MAX_STEER, WB, plot_car, BUBBLE_R
import planner.utils.reeds_shepp_path_planning as rs

# Local planner parameters
XY_GRID_RESOLUTION = 2 # meters per grid cell
YAW_GRID_RESOLUTION = np.deg2rad(10)  # radians per yaw step
SHOW_ANIMATION = True
DOWNSAMPLE_OBSTACLES_TO = 100  # max number of obstacles to keep for local planning
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
            PoseStamped, '/robot', self.pose_callback, 10
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
        self.res = None
        self.origin_x = None
        self.origin_y = None

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

    def _flip_xy(self, x: float, y: float):
        """Apply same flips as mapping node: I'=(W-1)-I, J'=(H-1)-J."""
        if self.map_msg is None:
            return None
        # print(f"self.msg: {self.map_msg}")
        w = self.map_msg.info.width
        h = self.map_msg.info.height
        return (x, (h - 1) - y )
    def pose_callback(self, msg: PoseStamped):
        fx, fy = float(msg.pose.position.x), float(msg.pose.position.y)
        if self.map_msg is not None:
            fx, fy = self._flip_xy(fx, fy)
            print(f"pose before: { fx, fy}")
            fx = fx *  self.res  +  self.res/2.0 #+ self.origin_x
            fy = fy *  self.res  +  self.res/2.0 #+ self.origin_y
            self.current_pose = [ fx, fy, np.arctan2(2*(msg.pose.orientation.w*msg.pose.orientation.z),
                                            1-2*msg.pose.orientation.z**2)]

        print(f"current_pose: {self.current_pose}")
                            
    # def map_callback(self, msg: OccupancyGrid):
    #     self.map_msg = msg
    #     # RESOLUTION IS ALREADY METERS/CELL — DO NOT DIVIDE BY 100
    #     self.res = float(msg.info.resolution)
    #     self.origin_x = float(msg.info.origin.position.x)
    #     self.origin_y = float(msg.info.origin.position.y)

    #     grid = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
    #     ys, xs = np.nonzero(grid > 50)  # occupied cells (indices)
    #     print(f"OG Map received with {len(xs)} obstacles sample points {xs[:5], ys[:5]}")

    #     # cells -> world meters (cell centers)
    #     ox_full = xs * self.res + self.origin_x + 0.5 * self.res
    #     oy_full = ys * self.res + self.origin_y + 0.5 * self.res

    #     print(f"OG map res={self.res}, origin=({self.origin_x:.3f}, {self.origin_y:.3f})")
    #     print(f"Map received with {len(ox_full)} obstacles sample (m) {ox_full[:5], oy_full[:5]}")

    #     if self.current_pose is None:
    #         self.ox = ox_full.tolist()
    #         self.oy = oy_full.tolist()
    #         return

    #     cx, cy = self.current_pose[0], self.current_pose[1]
    #     half_w, half_h = Local_Grid_Width / 2.0, Local_Grid_Height / 2.0

    #     # local window mask (meters)
    #     mask = np.logical_and.reduce((
    #         ox_full >= cx - half_w,
    #         ox_full <= cx + half_w,
    #         oy_full >= cy - half_h,
    #         oy_full <= cy + half_h
    #     ))
    #     ox_local = ox_full[mask]
    #     oy_local = oy_full[mask]

    #     min_x, max_x = cx - half_w, cx + half_w
    #     min_y, max_y = cy - half_h, cy + half_h

    #     print(f"Local planner window: x[{min_x:.2f}, {max_x:.2f}], y[{min_y:.2f}, {max_y:.2f}], "
    #         f"obstacles in window: {len(ox_local)}")

    #     # downsample if needed
    #     n_pts = len(ox_local)
    #     if n_pts > DOWNSAMPLE_OBSTACLES_TO:
    #         idx = np.random.choice(n_pts, DOWNSAMPLE_OBSTACLES_TO, replace=False)
    #         ox_local = ox_local[idx]
    #         oy_local = oy_local[idx]

    #     # ensure non-empty obstacle list for Config bounds
    #     if ox_local.size == 0:
    #         ox_local = np.array([cx - half_w, cx - half_w, cx + half_w, cx + half_w])
    #         oy_local = np.array([cy - half_h, cy + half_h, cy - half_h, cy + half_h])

    #     self.ox = ox_local.tolist()
    #     self.oy = oy_local.tolist()

    #     # choose local goal (meters)
    #     local_goal = self.get_local_goal()
    #     if local_goal is None:
    #         self.get_logger().warn("No local goal (global path empty or none in window)")
    #         return

    #     start = self.current_pose[:]   # [x,y,yaw] meters/rad
    #     goal  = local_goal

    #     if not (min_x <= start[0] <= max_x and min_y <= start[1] <= max_y):
    #         self.get_logger().warn("Start outside local window, skipping planning")
    #         return
    #     else:
    #         print(f"Local planner start inside window: {start}")
    #     if not (min_x <= goal[0] <= max_x and min_y <= goal[1] <= max_y):
    #         self.get_logger().warn("Goal outside local window, skipping planning")
    #         return
    #     else:
    #         print(f"Local planner goal inside window: {goal}")

    #     # -------------- critical bits below --------------

    #     # Keep planner grid aligned with the map resolution
    #     global XY_GRID_RESOLUTION
    #     XY_GRID_RESOLUTION = self.res

    #     # Add a 1-cell border around the window so the heuristic DP can’t index out of range
    #     frame_ox, frame_oy = self._build_window_frame(min_x, max_x, min_y, max_y, self.res)

    #     # Small margin around start/goal so they don’t quantize to the very last index
    #     margin = 2.0 * self.res
    #     sg_ox = [start[0]-margin, start[0]+margin, goal[0]-margin, goal[0]+margin]
    #     sg_oy = [start[1]-margin, start[1]+margin, goal[1]-margin, goal[1]+margin]

    #     # Augmented obstacles for the planner & distance heuristic
    #     ox_aug = self.ox + frame_ox + sg_ox
    #     oy_aug = self.oy + frame_oy + sg_oy

    #     self.get_logger().info(
    #         f"Local planner: start={start}, goal={goal}, obstacles={len(ox_aug)} (with frame+margin)"
    #     )

    #     # Plan with augmented obstacles
    #     self.plan_path_with_obstacles(start, goal, ox_aug, oy_aug)
    def _build_end_caps(self, min_x, max_x, min_y, max_y, res, pad_cells=1, stride_cells=3):
        """
        Two horizontal rails (front/back) sampled every stride_cells.
        Placed pad_cells outside the window to avoid interfering with planning.
        Returns (ox, oy) lists in meters.
        """
        pad   = pad_cells * res
        y_fwd = min_y - pad        # front rail (ahead)
        y_back= max_y + pad        # back rail (behind)

        xs = np.arange(min_x - pad, max_x + pad + 1e-9, stride_cells * res, dtype=float)

        ox = np.concatenate([xs, xs]).tolist()
        oy = np.concatenate([np.full_like(xs, y_fwd, dtype=float),
                            np.full_like(xs, y_back, dtype=float)]).tolist()
        return ox, oy
        
    def map_callback(self, msg: OccupancyGrid):
        self.map_msg = msg
        self.res = msg.info.resolution/100.0  # convert cm to m
        self.origin_x = msg.info.origin.position.x
        self.origin_y = msg.info.origin.position.y

        # convert occupancy grid to 2D numpy array
        grid = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        ys, xs = np.nonzero(grid > 50)
        print(f"OG Map received with {len(xs)} obstacles sample points {xs[:5], ys[:5]}")
      
        # # convert to world coords (cell centers)
        ox_full = xs *  self.res  +  self.res/2.0 #+  self.origin_x
        oy_full = ys *  self.res   +  self.res/2.0 #+  self.origin_y

        print(f"OG map res{self.res}, origin({self.origin_x}, {self.origin_y})")

        # ox_full, oy_full = self._flip_xy(xs,ys)

        print(f"Map received with {len(ox_full)} obstacles sample points {ox_full[:5], oy_full[:5]}")
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
            oy_full >= cy - half_h, #0.75 * Local_Grid_Height,
            oy_full <= cy + half_h #0.25 * Local_Grid_Height
        ))
        ox_local = ox_full[mask]
        oy_local = oy_full[mask]
        min_x = cx - half_w   # leftmost boundary of local window
        max_x = cx + half_w   # rightmost boundary
        min_y = cy - half_h #0.75 * Local_Grid_Height   # top boundary
        max_y = cy + half_h #0.25 * Local_Grid_Height   # bottom boundary

        print(f"Local planner window: x[{min_x}, {max_x}], y[{min_y}, {max_y}], obstacles in window: {len(ox_local)}")
       
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

        cap_ox, cap_oy = self._build_end_caps(min_x, max_x, min_y, max_y, XY_GRID_RESOLUTION,
                                      pad_cells=1,    # 1-cell outside
                                      stride_cells=3) # sparser sampling to “not affect planning a lot”

        # (optional) tiny margins so start/goal don’t quantize to the last index
        margin = 2.0 * XY_GRID_RESOLUTION
        sg_ox = [min_x-margin, max_x+margin]
        sg_oy = [min_y-margin, max_y+margin]

        # build augmented obstacle set used by the planner
        self.ox = ox_local.tolist()  + sg_ox #+ cap_ox
        self.oy = oy_local.tolist()  + sg_oy # + cap_oy 

        # self.ox = ox_local.tolist()
        # self.oy = oy_local.tolist()
        # print(f" Min and Max of ox: {min(self.ox), max(self.ox)}")
        # print(f" Min and Max of oy: {min(self.oy), max(self.oy)}")

        # choose local goal (furthest global path point inside window)
        local_goal = self.get_local_goal()
        if local_goal is None:
            self.get_logger().warn("No local goal (global path empty or none in window)")
            return

        start =  self.current_pose #* self.map_msg.info.resolution + self.map_msg.info.origin.position.x/2 , * self.map_msg.info.resolution + self.map_msg.info.origin.position.y/2
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
        # print(f"Planning from {start} to {goal} with {len(self.ox)} obstacles")
        # print(f" Min and Max of ox: Min X: {min(self.ox)}, Max X: {max(self.ox)}")
        # print(f" Min and Max of oy: Min Y: {min(self.oy)}, Max Y:{max(self.oy)}")
        path_result = hybrid_a_star_planning(start, goal, self.ox, self.oy, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)
        if path_result and len(path_result.x_list) > 0:
            self.publish_path(path_result)
            if SHOW_ANIMATION:
                self.plot_path(path_result)
    
    def publish_path(self, path_result):
        path_msg = Path()
        path_msg.header.frame_id = 'local_map'
        for x, y, yaw in zip(path_result.x_list, path_result.y_list, path_result.yaw_list):
            pose = PoseStamped()
            pose.pose.position.x = x + self.origin_x
            pose.pose.position.y = y + self.origin_y
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
            gx, gy = self._flip_xy(gx, gy)

            # print(f"Checking global path point before changing: ({gx}, {gy})")
            gx = gx * self.res + self.res/2.0 #+ self.origin_x 
            gy = gy * self.res + self.res/2.0 #+ self.origin_y 
            # print(f"Checking global path point after changing: ({gx}, {gy})")
            # Check if inside local window
            if (x_c - half_w <= gx <= x_c + half_w) and (y_c - half_h <= gy <= y_c + half_h):
                yaw = self.current_pose[2]  # optional: could compute heading from path
                self.get_logger().info(f"Local goal from global path: iter{self.iter}({gx}, {gy})")
                self.iter += 1
                return [gx, gy, yaw]

        # Fallback: last global point
        last_pose = self.global_path.poses[-1]
        print(f"Local planner goal: { self.current_pose}")
        print(f"Fallback to last global goal: ({last_pose.pose.position.x}, {last_pose.pose.position.y})")
        return [last_pose.pose.position.x, last_pose.pose.position.y, self.current_pose[2]]
    
    def _grid_to_world(self, ix: int, iy: int):
        assert self.map_msg is not None
        info = self.map_msg.info
        res = info.resolution
        ox0 = info.origin.position.x
        oy0 = info.origin.position.y
        h   = info.height

        # unflip Y
        # iy_nom = (h - 1) - iy

        x_w = ix * res + ox0 + res * 0.5
        y_w = iy * res + oy0 + res * 0.5
        return float(x_w), float(y_w)
    
    def _build_window_frame(self, min_x, max_x, min_y, max_y, step):
        xs = np.arange(min_x, max_x + 1e-9, step, dtype=float)
        ys = np.arange(min_y, max_y + 1e-9, step, dtype=float)

        # top & bottom edges
        fx = np.concatenate([xs, xs])
        fy = np.concatenate([np.full_like(xs, min_y), np.full_like(xs, max_y)])

        # left & right edges (avoid duplicating corners)
        if len(ys) > 2:
            fy2 = ys[1:-1]
            fx2 = np.concatenate([np.full_like(fy2, min_x), np.full_like(fy2, max_x)])
            fy2 = np.concatenate([fy2, fy2])
            fx = np.concatenate([fx, fx2])
            fy = np.concatenate([fy, fy2])
        return fx.tolist(), fy.tolist()

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
