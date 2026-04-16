import json
import threading
import time

import gym
import numpy as np
import roslibpy


class PythGazeboParking(gym.Env):

    metadata = {}

    max_episode_steps = 500

    # 如果之后要改，这里也要同步修改
    OBS_DIM = 43
    ACT_LOW  = np.array([-0.01,  -0.03], dtype=np.float32)
    ACT_HIGH = np.array([0.01,   0.03], dtype=np.float32)

    def __init__(
        self,
        n_obstacles: int = 0,
        bridge_env_id: str = "sampler",
        rosbridge_host: str = "localhost",
        rosbridge_port: int = 9090,
        timeout: float = 30.0,
        **kwargs,
    ):
        super().__init__()

        self.n_obstacles = n_obstacles
        self.bridge_env_id = bridge_env_id
        self.timeout = timeout

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=self.ACT_LOW, high=self.ACT_HIGH, dtype=np.float32
        )

        self.additional_info = {
        "constraint": {"shape": (1,), "dtype": np.float32}
    }

        # rosbridge 连接
        self._client = roslibpy.Ros(host=rosbridge_host, port=rosbridge_port)
        self._client.run()
        print(f"[Wrapper] rosbridge 连接成功: {rosbridge_host}:{rosbridge_port}")

        # 发布者
        self._pub_action = roslibpy.Topic(
            self._client, "/rl/action", "std_msgs/String"
        )
        self._pub_reset = roslibpy.Topic(
            self._client, "/rl/reset", "std_msgs/String"
        )

        # 订阅者 + 事件
        self._step_result  = None
        self._reset_result = None
        self._step_event   = threading.Event()
        self._reset_event  = threading.Event()

        self._sub_result = roslibpy.Topic(
            self._client, "/rl/result", "std_msgs/String"
        )
        self._sub_result.subscribe(self._step_cb)

        self._sub_reset = roslibpy.Topic(
            self._client, "/rl/reset_result", "std_msgs/String"
        )
        self._sub_reset.subscribe(self._reset_cb)

    # 回调
    def _step_cb(self, msg):
        data = json.loads(msg["data"])
        msg_env_id = data.get("env_id", "sampler")
        if msg_env_id != self.bridge_env_id:
            return
        self._step_result = data
        self._step_event.set()

    def _reset_cb(self, msg):
        data = json.loads(msg["data"])
        msg_env_id = data.get("env_id", "sampler")
        if msg_env_id != self.bridge_env_id:
            return
        self._reset_result = data
        self._reset_event.set()

    # gym 接口
    def reset(self, **kwargs):
        self._reset_event.clear()
        payload = json.dumps({
            "env_id": self.bridge_env_id,
            "n_obstacles": self.n_obstacles,
        })
        self._pub_reset.publish(roslibpy.Message({"data": payload}))

        if not self._reset_event.wait(timeout=self.timeout):
            raise TimeoutError("[Wrapper] reset 超时，检查 gazebo_bridge 是否在运行")

        obs = np.array(self._reset_result["obs"], dtype=np.float32)
        info = {
            "constraint": np.array([0.0]),
            "_bridge_reset_seq": int(self._reset_result.get("reset_seq", 0)),
        }
        return obs, info

    def step(self, action):
        action = np.clip(action, self.ACT_LOW, self.ACT_HIGH)

        self._step_event.clear()
        payload = json.dumps({
            "env_id": self.bridge_env_id,
            "action": action.tolist(),
        })
        self._pub_action.publish(roslibpy.Message({"data": payload}))

        if not self._step_event.wait(timeout=self.timeout):
            raise TimeoutError("[Wrapper] step 超时，检查 gazebo_bridge 是否在运行")

        r = self._step_result
        obs    = np.array(r["obs"], dtype=np.float32)
        reward = float(r["reward"])
        done   = bool(r["done"])
        info = r.get("info", {})
        if "constraint" in info:
            info["constraint"] = np.array(info["constraint"], dtype=np.float32)
        if "_bridge_reset_seq" in info:
            info["_bridge_reset_seq"] = int(info["_bridge_reset_seq"])
        return obs, reward, done, info

    def close(self):
        if hasattr(self, "_sub_result"):
            self._sub_result.unsubscribe()
        if hasattr(self, "_sub_reset"):
            self._sub_reset.unsubscribe()
        # roslibpy uses Twisted's global reactor; terminating it here makes
        # subsequent Ros.run() calls in the same process fail with
        # ReactorNotRestartable. For transient env instances (e.g., init_args),
        # just close websocket if available and keep reactor alive.
        if hasattr(self._client, "close"):
            self._client.close()

    def seed(self, seed=None):
        return []



def env_creator(**kwargs):
    return PythGazeboParking(**kwargs)
