#!/usr/bin/env python3


import rospy
import numpy as np
import gym
from gym import spaces
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ContactsState
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
import tf.transformations as tft


import sys
sys.path.append('/home/lin/ros_rl_ws/src/my_env_generator/scripts')
from env_generator import EnvGenerator

 
def angle_normalize(x):

    return ((x + np.pi) % (2 * np.pi)) - np.pi


class GazeboEnv(gym.Env):
    """
    观测空间（41维)：
      [0]    v_norm          当前线速度归一化
      [1]    w_norm          当前角速度归一化
      [2:38] lidar×36        雷达降采样（360→36），归一化到[0,1]
      [38]   rel_dist        到车库距离，归一化
      [39]   rel_angle       车库方向相对车头角度
      [40]   heading_err     朝向误差

    动作空间（2维)：
      [0]  Δv   线速度增量
      [1]  Δω   角速度增量
    """

    # 车辆参数
    R_MIN = 1.2
    V_MAX       = 0.3 # 实车最大速度约为3.5
    W_MAX       = V_MAX / R_MIN
    V_DELTA_MAX = 0.1
    W_DELTA_MAX = 0.3

    # 雷达参数
    N_LIDAR_BEAMS   = 36
    LIDAR_MAX_RANGE = 10.0

    # 最大步数
    MAX_STEPS = 240



    def __init__(self):
        super().__init__()

        # 初始化 ROS 节点（如果还没有）
        if not rospy.get_node_uri():
            rospy.init_node('gazebo_env', anonymous=True)

        self.env_gen = EnvGenerator()

        # 动作空间：[Δv, Δω]
        self.dt = 0.1
        lb = np.array([-self.V_DELTA_MAX * self.dt,
                       -self.W_DELTA_MAX * self.dt], dtype=np.float32)
        hb = np.array([ self.V_DELTA_MAX * self.dt,
                        self.W_DELTA_MAX * self.dt], dtype=np.float32)
        self.action_space = spaces.Box(low=lb, high=hb, dtype=np.float32)

        # 观测空间：41维，归一化到[-1,1]
        obs_dim = 2 + self.N_LIDAR_BEAMS + 3
        self.observation_space = spaces.Box(
            low  = np.full(obs_dim, -1.0, dtype=np.float32),
            high = np.full(obs_dim,  1.0, dtype=np.float32),
        )

        # 当前状态
        self.current_v   = 0.0
        self.current_w   = 0.0
        self.oddset      = 0.15 # 里程计与车辆中心位置的偏移，实际为0.255，但是太过困难，适当放宽一些
        self.goal_x      = 0.0
        self.goal_y      = -2.0 - self.oddset
        self.goal_theta  = np.pi / 2
        self._prev_dist  = 0.0
        self.steps       = 0
        self.collision = False

        # 传感器缓存
        self._scan_data  = None
        self._odom_data  = None
        self._lidar_norm = np.ones(self.N_LIDAR_BEAMS)  # 归一化

        # ROS 订阅
        rospy.Subscriber('/scan', LaserScan, self._scan_cb)
        rospy.Subscriber('/odom', Odometry,  self._odom_cb)

        # ROS 发布
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # Gazebo 控制服务
        rospy.wait_for_service('/gazebo/pause_physics')
        rospy.wait_for_service('/gazebo/unpause_physics')
        self.pause   = rospy.ServiceProxy('/gazebo/pause_physics',  Empty)
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)

        # 订阅碰撞检测话题
        self.contact_sub = rospy.Subscriber('/robot_contacts', ContactsState, self._contact_cb)

        rospy.loginfo("GazeboEnv 初始化完成")


    def reset(self, n_obstacles=3):
   
        self.steps      = 0
        self.current_v  = 0.0
        self.current_w  = 0.0

        stop = Twist()
        self.cmd_pub.publish(stop)
        

        # 暂停物理，安全布置场景
        self.pause()
        self.goal_x, self.goal_y = self.env_gen.reset_env(n_obstacles)
        self.unpause()
        rospy.sleep(0.5)

        self.collision  = False

        obs = self._get_obs()

        # 记录初始距离（用于势能奖励）
        x, y, _ = self._get_pose()
        self._prev_dist = np.hypot(x - self.goal_x, y - self.goal_y)

        self.pause() 

        return obs

    def step(self, action):
        """
        返回: obs, reward, done, info
        """
        self.steps += 1

        # 速度增量累加
        delta_v, delta_w = action
        self.current_v = np.clip(self.current_v + delta_v,
                                 -self.V_MAX, self.V_MAX)
        self.current_w = np.clip(self.current_w + delta_w,
                                 -self.W_MAX, self.W_MAX)
        
        v_min = min(self.R_MIN * abs(self.current_w), self.V_MAX)
        if self.current_v >= 0:
            self.current_v = max(self.current_v, v_min)
        elif self.current_v < 0:
            self.current_v = min(self.current_v, -v_min)
                
        self.unpause()

        # 发布控制指令
        cmd = Twist()
        cmd.linear.x  = float(self.current_v)
        cmd.angular.z = float(self.current_w)
        self.cmd_pub.publish(cmd)

        # 等待一个控制周期
        rospy.sleep(self.dt)

        self.pause()

        obs    = self._get_obs()
        reward = self._compute_reward(action)
        done   = False
        info   = {}
        info["constraint"] = np.array([0.0])

        # 碰撞惩罚和成功奖励
        if self._check_collision():
            reward -= 200.0
            done    = True
            info["constraint"] = np.array([1.0])

        if self._check_success():
            reward += 500.0
            done    = True

        if self.steps >= self.MAX_STEPS:
            info["TimeLimit.truncated"] = True
            done = True

        return obs, reward, done, info


    def _compute_reward(self, action) -> float:
        x, y, theta = self._get_pose()
        delta_v, delta_w = action

        distance = np.hypot(x - self.goal_x, y - self.goal_y)

        # 势能奖励：靠近车库给正奖励
        r_potential = (self._prev_dist - distance) * 10.0
        self._prev_dist = distance

        # 朝向奖励：仅在接近车库时生效
        if distance < 0.7:
            r_heading = -abs(angle_normalize(theta - self.goal_theta)) * 1.0
        else:
            r_heading = 0.0

        # 步数惩罚
        r_step = -0.05

        # 动作平滑惩罚
        r_smooth = -(delta_v ** 2 + delta_w ** 2) * 2.0

        # 雷达安全惩罚
        min_lidar_m = float(np.min(self._lidar_norm)) * self.LIDAR_MAX_RANGE
        safety_thresh = 0.5
        r_safety = min(0.0, (min_lidar_m - safety_thresh) * 10.0)

        # 换向惩罚
        v_prev = self.current_v - delta_v
        if (v_prev > 0 > self.current_v) or (v_prev < 0 < self.current_v):
            r_direction = -0.3
        else:
            r_direction = 0.0

        return r_potential + r_heading + r_step + r_smooth + r_safety + r_direction


    #  结束条件
    def _check_collision(self) -> bool:
        return self.collision

    def _check_success(self) -> bool:
        """判断是否成功入库"""
        x, y, theta = self._get_pose()
        dist = np.hypot(x - self.goal_x, y - self.goal_y)
        aerr = abs(angle_normalize(theta - self.goal_theta))
        angle_ok = aerr < 0.07 or abs(aerr - np.pi) < 0.07
        return (dist < 0.1 and
                abs(self.current_v) < 0.02 and
                abs(self.current_w) < 0.02 and
                angle_ok)


    def _get_obs(self) -> np.ndarray:

        self._wait_for_sensors()

        x, y, theta = self._get_pose()

        # 速度归一化
        v_norm = np.clip(self.current_v / self.V_MAX, -1.0, 1.0)
        w_norm = np.clip(self.current_w / self.W_MAX, -1.0, 1.0)

        # 雷达降采样并归一化（360→36）
        raw = np.array(self._scan_data.ranges, dtype=np.float32)
        raw = np.clip(raw, 0.0, self.LIDAR_MAX_RANGE)
        raw[np.isinf(raw)] = self.LIDAR_MAX_RANGE
        # 每隔10个取一个（360/36=10）
        downsampled = raw[::10][:self.N_LIDAR_BEAMS]
        self._lidar_norm = downsampled / self.LIDAR_MAX_RANGE

        # 目标相对信息
        dx = self.goal_x - x
        dy = self.goal_y - y
        rel_dist    = np.clip(np.hypot(dx, dy) / self.LIDAR_MAX_RANGE,
                              0.0, 1.0)
        rel_angle   = angle_normalize(np.arctan2(dy, dx) - theta) / np.pi
        heading_err = angle_normalize(self.goal_theta - theta) / np.pi

        return np.concatenate([
            [v_norm, w_norm],
            self._lidar_norm,
            [rel_dist, rel_angle, heading_err],
        ]).astype(np.float32)


    #  传感器读取工具方法

    def _wait_for_sensors(self):
        """等待传感器数据到位"""
        rate = rospy.Rate(100)
        while self._scan_data is None or self._odom_data is None:
            rate.sleep()

    def _get_pose(self):
        """从里程计提取 (x, y, theta)"""
        pos = self._odom_data.pose.pose.position
        q   = self._odom_data.pose.pose.orientation
        _, _, theta = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
        return pos.x, pos.y, theta

    def _get_xy(self):
        pos = self._odom_data.pose.pose.position
        return pos.x, pos.y

    def _scan_cb(self, msg):
        self._scan_data = msg

    def _odom_cb(self, msg):
        self._odom_data = msg

    def _contact_cb(self, msg):
        for state in msg.states:
            col1 = state.collision1_name
            col2 = state.collision2_name

            # 只关心 base_link 的碰撞
            if 'base_link' not in col1 and 'base_link' not in col2:
                continue

            self.collision = True
            return


# ================================================================
#  单独测试入口
# ================================================================
if __name__ == '__main__':
    rospy.init_node('gazebo_env_test')

    env = GazeboEnv()

    rospy.loginfo("=== 测试 reset ===")
    obs = env.reset(n_obstacles=7)
    rospy.loginfo(f"obs shape: {obs.shape}")   # 期望 (41,)
    rospy.loginfo(f"obs: {obs}")

    rospy.loginfo("=== 测试 step ===")
    for i in range(1000):
        # action = env.action_space.sample()  # 随机动作，测试用
        action = np.array([0, 0.03])
        rospy.loginfo(f"step {i}: action={action}")
        obs, reward, done, info = env.step(action)
        x, y, theta = env._get_pose()
        rospy.loginfo(f"step {i}: pose=({x:.2f}, {y:.2f}, {theta:.2f}), reward={reward:.3f}, done={done}")
        rospy.loginfo(f"step {i}: obs={[env.current_v, env.current_w, obs[39:]]}")

        if done:
            obs = env.reset(n_obstacles=7)

    rospy.loginfo("=== 测试完成 ===")
