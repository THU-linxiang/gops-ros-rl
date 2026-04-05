#!/usr/bin/env python3
"""
把 GazeboEnv 的 step/reset 通过 ROS 话题暴露给 GOPS 训练进程
"""

import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_GEN_DIR = os.path.abspath(
    os.path.join(CURRENT_DIR, "..", "..", "my_env_generator", "scripts")
)
if ENV_GEN_DIR not in sys.path:
    sys.path.append(ENV_GEN_DIR)

import json
import threading
import numpy as np
import rospy
from std_msgs.msg import String

from gazebo_env import GazeboEnv


class GazeboBridge:
    def __init__(self):
        rospy.init_node("gazebo_bridge", anonymous=False)

        self.env = GazeboEnv()
        self.n_obstacles = rospy.get_param("~n_obstacles", 3)

        # 发布：每个step之后的 obs/reward/done/info 
        self.pub_result = rospy.Publisher("/rl/result", String, queue_size=1)
        # 发布：reset 之后的初始 obs
        self.pub_reset_result = rospy.Publisher("/rl/reset_result", String, queue_size=1)

        # Gazebo step/reset must be serialized across sampler/evaluator requests.
        self._env_lock = threading.Lock()

        # 订阅：GOPS 发来的 action
        rospy.Subscriber("/rl/action", String, self._action_cb)
        # 订阅：GOPS 发来的 reset 请求
        rospy.Subscriber("/rl/reset", String, self._reset_cb)

        rospy.loginfo("[Bridge] 初始化完成，等待 GOPS 连接...")
        rospy.spin()


    def _action_cb(self, msg):
        """收到 action，执行一步，把结果发回去"""
        try:
            data = json.loads(msg.data)
            if isinstance(data, dict):
                env_id = data.get("env_id", "sampler")
                action_data = data.get("action", [])
            else:
                env_id = "sampler"
                action_data = data

            action = np.array(action_data, dtype=np.float32)
            with self._env_lock:
                obs, reward, done, info = self.env.step(action)

            result = json.dumps({
                "env_id": env_id,
                "obs":    obs.tolist(),
                "reward": float(reward),
                "done":   bool(done),
                "info":   {k: v.tolist() if isinstance(v, np.ndarray) else
                            (bool(v) if isinstance(v, (bool, np.bool_)) else v)
                        for k, v in info.items()}
            })
            self.pub_result.publish(result)

        except Exception as e:
            rospy.logerr(f"[Bridge] action_cb 异常: {e}")

    def _reset_cb(self, msg):
        """收到 reset 请求，重置环境，把初始 obs 发回去"""
        try:
            data = json.loads(msg.data) if msg.data else {}
            env_id = data.get("env_id", "sampler")
            n_obstacles = data.get("n_obstacles", self.n_obstacles)

            with self._env_lock:
                obs = self.env.reset(n_obstacles=n_obstacles)

            result = json.dumps({
                "env_id": env_id,
                "obs": obs.tolist(),
            })
            self.pub_reset_result.publish(result)

        except Exception as e:
            rospy.logerr(f"[Bridge] reset_cb 异常: {e}")


if __name__ == "__main__":
    GazeboBridge()

