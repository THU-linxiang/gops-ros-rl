#!/usr/bin/env python3
import rospy
from nav_msgs.msg import Odometry
import tf.transformations as tft

def odom_cb(msg):
    pos = msg.pose.pose.position
    q   = msg.pose.pose.orientation
    
    _, _, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
    
    rospy.loginfo(f"x={pos.x:.3f}  y={pos.y:.3f}  yaw={yaw:.3f} rad ({yaw*57.3:.1f} deg)")

rospy.init_node("odom_printer")
rospy.Subscriber("/odom", Odometry, odom_cb)

rate = rospy.Rate(0.01)  # 1Hz
while not rospy.is_shutdown():
    rate.sleep()
