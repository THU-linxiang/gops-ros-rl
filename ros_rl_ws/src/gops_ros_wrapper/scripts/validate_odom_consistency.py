#!/usr/bin/env python3

import json
import math
import time

import rospy
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from std_msgs.msg import String
import tf.transformations as tft


def angle_normalize(x):
    return ((x + math.pi) % (2.0 * math.pi)) - math.pi


def yaw_from_quat(q):
    return tft.euler_from_quaternion([q.x, q.y, q.z, q.w])[2]


class OdomConsistencyValidator:
    def __init__(self):
        self.robot_model_name = rospy.get_param("~robot_model_name", "my_robot")
        self.duration = float(rospy.get_param("~duration", 10.0))
        self.sample_hz = float(rospy.get_param("~sample_hz", 30.0))
        self.pos_tol = float(rospy.get_param("~pos_tol", 0.25))
        self.yaw_tol = float(rospy.get_param("~yaw_tol", 0.25))
        self.do_reset_test = bool(rospy.get_param("~do_reset_test", True))
        self.reset_timeout = float(rospy.get_param("~reset_timeout", 2.0))

        self.odom_msg = None
        self.model_states = None
        self.last_reset_result = None

        self.odom_sub = rospy.Subscriber("/odom", Odometry, self._odom_cb)
        self.model_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self._model_cb)
        self.reset_sub = rospy.Subscriber("/rl/reset_result", String, self._reset_result_cb)
        self.reset_pub = rospy.Publisher("/rl/reset", String, queue_size=1)

    def _odom_cb(self, msg):
        self.odom_msg = msg

    def _model_cb(self, msg):
        self.model_states = msg

    def _reset_result_cb(self, msg):
        self.last_reset_result = msg

    def _get_robot_model_pose(self):
        if self.model_states is None:
            return None
        try:
            idx = self.model_states.name.index(self.robot_model_name)
        except ValueError:
            return None
        return self.model_states.pose[idx]

    def _compute_errors(self):
        if self.odom_msg is None:
            return None
        model_pose = self._get_robot_model_pose()
        if model_pose is None:
            return None

        odom_pose = self.odom_msg.pose.pose
        dx = odom_pose.position.x - model_pose.position.x
        dy = odom_pose.position.y - model_pose.position.y
        pos_err = math.hypot(dx, dy)

        odom_yaw = yaw_from_quat(odom_pose.orientation)
        model_yaw = yaw_from_quat(model_pose.orientation)
        yaw_err = abs(angle_normalize(odom_yaw - model_yaw))
        return pos_err, yaw_err

    def wait_for_topics(self, timeout=5.0):
        t0 = time.time()
        rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            if self.odom_msg is not None and self._get_robot_model_pose() is not None:
                return True
            if time.time() - t0 > timeout:
                return False
            rate.sleep()
        return False

    def run_continuous_check(self):
        samples = 0
        pos_errors = []
        yaw_errors = []

        rate = rospy.Rate(self.sample_hz)
        t_end = time.time() + self.duration
        while not rospy.is_shutdown() and time.time() < t_end:
            e = self._compute_errors()
            if e is not None:
                pos_errors.append(e[0])
                yaw_errors.append(e[1])
                samples += 1
            rate.sleep()

        if samples == 0:
            return None

        result = {
            "samples": samples,
            "pos_mean": sum(pos_errors) / samples,
            "pos_max": max(pos_errors),
            "yaw_mean": sum(yaw_errors) / samples,
            "yaw_max": max(yaw_errors),
        }
        return result

    def run_reset_freshness_check(self):
        # If bridge is not connected, report skip with actionable hint.
        if self.reset_pub.get_num_connections() == 0:
            return None, "no subscriber on /rl/reset (start gazebo_bridge)"

        if self.odom_msg is None:
            return False, "no odom before reset"

        prev_seq = self.odom_msg.header.seq
        self.last_reset_result = None
        payload = json.dumps({"env_id": "validator", "n_obstacles": 3})
        self.reset_pub.publish(String(data=payload))

        t0 = time.time()
        got_reset_reply = False
        got_new_odom = False

        rate = rospy.Rate(200)
        while not rospy.is_shutdown() and (time.time() - t0) < self.reset_timeout:
            if self.last_reset_result is not None:
                try:
                    data = json.loads(self.last_reset_result.data)
                    if data.get("env_id") == "validator":
                        got_reset_reply = True
                except Exception:
                    pass
            if self.odom_msg is not None and self.odom_msg.header.seq > prev_seq:
                got_new_odom = True

            if got_reset_reply and got_new_odom:
                return True, "reset reply received and odom advanced"
            rate.sleep()

        if not got_reset_reply:
            return False, "no reset reply from bridge (check gazebo_bridge)"
        if not got_new_odom:
            return False, "odom did not advance after reset"
        return True, "ok"


def main():
    rospy.init_node("validate_odom_consistency", anonymous=True)
    validator = OdomConsistencyValidator()

    rospy.loginfo("[check] waiting for /odom and /gazebo/model_states ...")
    if not validator.wait_for_topics(timeout=8.0):
        rospy.logerr("[FAIL] topics not ready: /odom or /gazebo/model_states")
        return

    cont = validator.run_continuous_check()
    if cont is None:
        rospy.logerr("[FAIL] no samples collected for continuous check")
        return

    rospy.loginfo(
        "[check] samples=%d pos_mean=%.4f pos_max=%.4f yaw_mean=%.4f yaw_max=%.4f",
        cont["samples"], cont["pos_mean"], cont["pos_max"], cont["yaw_mean"], cont["yaw_max"]
    )

    pass_cont = (cont["pos_max"] <= validator.pos_tol) and (cont["yaw_max"] <= validator.yaw_tol)
    if not pass_cont:
        rospy.logerr("[FAIL] odom/model pose mismatch exceeds tolerance")
    else:
        rospy.loginfo("[PASS] odom/model pose consistency within tolerance")

    if validator.do_reset_test:
        ok, msg = validator.run_reset_freshness_check()
        if ok is None:
            rospy.logwarn("[SKIP] reset freshness check: %s", msg)
        elif ok:
            rospy.loginfo("[PASS] reset freshness check: %s", msg)
        else:
            rospy.logerr("[FAIL] reset freshness check: %s", msg)


if __name__ == "__main__":
    main()
