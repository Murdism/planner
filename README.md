# planner
D* lite for global planning and A* for local planning 


## Installation

### 1. Create ROS2 Workspace
```bash
mkdir ~/planner_ws
cd ~/planner_ws
mkdir src
cd src
```

### 2. Clone Repository
```bash
git clone --depth 1 https://github.com/Murdism/planner.git .
```

### 3. Build and Source
```bash
cd ~/planner_ws
colcon build
source install/setup.bash
```


## Usage

### Running Individual Nodes
***Note* To run planners seperatly

#### Global Node
```bash
ros2 run planner global_planner 
```

#### Local Node
```bash
ros2 run planner local_planner 
```