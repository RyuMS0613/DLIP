import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
import time
import math


BASIC_POSE = [0, 20, -30, 0, 10, 90]

class PiperController(Node):
    def __init__(self):
        super().__init__('piper_controller')
        self.enable_pub = self.create_publisher(Bool, '/enable_flag', 10)
        self.joint_pub  = self.create_publisher(JointState, '/joint_states', 10)
        time.sleep(1.0)

    def enable(self):
        msg = Bool()
        msg.data = True
        self.enable_pub.publish(msg)
        self.get_logger().info("Enable Robot!")
        time.sleep(1.0)

    def move_to(self, j1=0.0, j2=0.0, j3=0.0,
                j4=0.0, j5=0.0, j6=0.0,
                gripper=0.0, speed=0.3):

        msg = JointState()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'piper_single'

        # ✅ 8개로 수정 (joint8 추가)
        msg.name = ['joint1','joint2','joint3',
                    'joint4','joint5','joint6',
                    'joint7','joint8']

        msg.position = [
            math.radians(j1),
            math.radians(j2),
            math.radians(j3),
            math.radians(j4),
            math.radians(j5),
            math.radians(j6),
            gripper,   # joint7 (그리퍼)
            0.0        # joint8 ✅ 추가
        ]
        msg.velocity = [speed] * 8   # ✅ 8개
        msg.effort   = [0.0] * 7 + [0.2]  # ✅ 8개

        self.joint_pub.publish(msg)
        self.get_logger().info(
            f"Move → J1:{j1}° J2:{j2}° J3:{j3}° "
            f"J4:{j4}° J5:{j5}° J6:{j6}°"
        )

def main():
    rclpy.init()
    robot = PiperController()

    robot.enable()

    print("=== Setup ===")
    robot.move_to(0, 0, 0, 0, 0, 0)
    time.sleep(3.0)

    print("=== Move Joint ===")
    # j1 - Base / j2, j3, j5 = Center 
    robot.move_to(*BASIC_POSE, speed=0.3)
    time.sleep(3.0)

    print("=== Return ===")
    robot.move_to(0, 0, 0, 0, 0, 0)
    time.sleep(3.0)

    print("Complete!")
    robot.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()