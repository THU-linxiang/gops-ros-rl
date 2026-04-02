#!/usr/bin/env python3

import rospy
import random
import numpy as np
from gazebo_msgs.srv import SpawnModel, DeleteModel, SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Point, Quaternion
from shapely.geometry import Point as ShapelyPoint, Polygon


class CircleObstacle:
    def __init__(self, cx, cy, radius):
        self.cx = cx
        self.cy = cy
        self.radius = radius

    def to_shapely(self):
        """用于生成时的重叠预检，不参与训练时碰撞判断"""
        return ShapelyPoint(self.cx, self.cy).buffer(self.radius)

    def to_sdf(self, name):
        height = 0.6
        return f"""
        <sdf version='1.6'>
          <model name='{name}'>
            <static>true</static>
            <link name='link'>
              <collision name='col'>
                <geometry>
                  <cylinder>
                    <radius>{self.radius}</radius>
                    <length>{height}</length>
                  </cylinder>
                </geometry>
              </collision>
              <visual name='vis'>
                <geometry>
                  <cylinder>
                    <radius>{self.radius}</radius>
                    <length>{height}</length>
                  </cylinder>
                </geometry>
                <material><script><n>Gazebo/Blue</n></script></material>
              </visual>
            </link>
          </model>
        </sdf>"""

    def spawn_pose(self):
        return Pose(position=Point(x=self.cx, y=self.cy, z=0.3))


class BoxObstacle:
    def __init__(self, cx, cy, half_w, half_h, angle):
        self.cx = cx
        self.cy = cy
        self.half_w = half_w
        self.half_h = half_h
        self.angle = angle

    def _corners(self):
        offsets = np.array([
            [-self.half_w, -self.half_h],
            [ self.half_w, -self.half_h],
            [ self.half_w,  self.half_h],
            [-self.half_w,  self.half_h],
        ])
        c, s = np.cos(self.angle), np.sin(self.angle)
        R = np.array([[c, -s], [s, c]])
        return offsets.dot(R.T) + np.array([self.cx, self.cy])

    def to_shapely(self):
        return Polygon(self._corners())

    def to_sdf(self, name):
        sx = self.half_w * 2
        sy = self.half_h * 2
        sz = random.uniform(0.3, 0.8)
        return f"""
        <sdf version='1.6'>
          <model name='{name}'>
            <static>true</static>
            <link name='link'>
              <collision name='col'>
                <geometry>
                  <box><size>{sx} {sy} {sz}</size></box>
                </geometry>
              </collision>
              <visual name='vis'>
                <geometry>
                  <box><size>{sx} {sy} {sz}</size></box>
                </geometry>
                <material><script><n>Gazebo/Orange</n></script></material>
              </visual>
            </link>
          </model>
        </sdf>"""

    def spawn_pose(self):
        qz = np.sin(self.angle / 2.0)
        qw = np.cos(self.angle / 2.0)
        return Pose(
            position=Point(x=self.cx, y=self.cy, z=0.3),
            orientation=Quaternion(x=0.0, y=0.0, z=qz, w=qw)
        )


class EnvGenerator:

    FIELD_X_MIN = -3.0
    FIELD_X_MAX =  3.0
    FIELD_Y_MIN = -3.0
    FIELD_Y_MAX =  3.0

    GARAGE_X =  0.0
    GARAGE_Y = -2.0


    def __init__(self):
        rospy.loginfo("等待Gazebo服务启动...")
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        rospy.wait_for_service('/gazebo/delete_model')
        rospy.wait_for_service('/gazebo/set_model_state')

        self.spawn_srv  = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.delete_srv = rospy.ServiceProxy('/gazebo/delete_model',    DeleteModel)
        self.set_state  = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

        self.obstacle_names = []
        rospy.loginfo("EnvGenerator 初始化完成")

    def reset_env(self, n_obstacles=5):
    # 每次 reset 时重新采样起始位置
        self.ROBOT_START_X = np.random.uniform(
            self.FIELD_X_MIN + 0.7, self.FIELD_X_MAX - 0.7
        )
        if self.GARAGE_X - 1.0 <= self.ROBOT_START_X <= self.GARAGE_X + 1.0:
            self.ROBOT_START_Y = np.random.uniform(
                self.GARAGE_Y + 1.0, self.FIELD_Y_MAX - 0.7
            )
        else:
            self.ROBOT_START_Y = np.random.uniform(
                self.FIELD_Y_MIN + 0.7, self.FIELD_Y_MAX - 0.7
            )

        self.ROBOT_START_YAW = np.random.uniform(0, 2 * np.pi)

        self._clear_obstacles()
        self._spawn_obstacles(n_obstacles)  
        self._reset_robot()
        return self.GARAGE_X, self.GARAGE_Y


    def _clear_obstacles(self):
        for name in self.obstacle_names:
            try:
                self.delete_srv(name)
            except Exception as e:
                rospy.logwarn(f"删除障碍物 {name} 失败: {e}")
        self.obstacle_names = []

    def _spawn_obstacles(self, n):
        placed = []
        for i in range(n):
            for _ in range(50):
                ox = np.random.uniform(self.FIELD_X_MIN + 0.5,
                                       self.FIELD_X_MAX - 0.5)
                oy = np.random.uniform(self.FIELD_Y_MIN + 1.0,
                                       self.FIELD_Y_MAX - 0.5)

                # 车辆起点安全距离
                if np.hypot(ox - self.ROBOT_START_X,
                            oy - self.ROBOT_START_Y) < 1.0:
                    continue

                # 车库入口禁区
                if (abs(ox - self.GARAGE_X) < 0.6 and
                        self.GARAGE_Y < oy < self.GARAGE_Y + 1.5):
                    continue

                # 车库本体禁区
                if (abs(ox - self.GARAGE_X) < 0.6 and
                        abs(oy - self.GARAGE_Y) < 0.6):
                    continue

                # 随机类型：圆形或矩形
                if np.random.rand() < 0.5:
                    r = np.random.uniform(0.15, 0.25)
                    candidate = CircleObstacle(ox, oy, r)
                else:
                    hw  = np.random.uniform(0.10, 0.25)
                    hh  = np.random.uniform(0.10, 0.25)
                    ang = np.random.uniform(0, np.pi)
                    candidate = BoxObstacle(ox, oy, hw, hh, ang)

                # 生成时预检：避免障碍物互相堆叠
                cshape = candidate.to_shapely()
                if any(cshape.intersects(ob.to_shapely()) for ob in placed):
                    continue

                name = f"obstacle_{i}"
                self._spawn_single(name, candidate)
                placed.append(candidate)
                break

    def _spawn_single(self, name, obstacle):
        sdf  = obstacle.to_sdf(name)
        pose = obstacle.spawn_pose()
        try:
            self.spawn_srv(name, sdf, '', pose, 'world')
            self.obstacle_names.append(name)
        except Exception as e:
            rospy.logerr(f"生成障碍物 {name} 失败: {e}")

    def _reset_robot(self):
        qz = np.sin(self.ROBOT_START_YAW / 2.0)
        qw = np.cos(self.ROBOT_START_YAW / 2.0)
        state = ModelState()
        state.model_name = 'my_robot'
        state.pose = Pose(
            position=Point(x=self.ROBOT_START_X,
                           y=self.ROBOT_START_Y,
                           z=0),
            orientation=Quaternion(x=0, y=0, z=qz, w=qw)
        )
        state.twist.linear.x  = 0
        state.twist.linear.y  = 0
        state.twist.angular.z = 0
        state.reference_frame = 'world'
        try:
            self.set_state(state)
        except Exception as e:
            rospy.logerr(f"重置车辆位置失败: {e}")


if __name__ == '__main__':
    rospy.init_node('env_generator_test')
    gen = EnvGenerator()

    # for i in range(1, 2):
    #     rospy.loginfo(f"=== 测试第{i}次reset ===")
    #     goal = gen.reset_env(n_obstacles = 5)
    #     rospy.loginfo(f"目标位置: {goal}")
    #     rospy.sleep(4.0)

    # rospy.loginfo("=== 测试完成 ===")
