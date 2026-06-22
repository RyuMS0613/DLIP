#!/usr/bin/env python3
"""
PiPER Robot - End-Effector Cartesian Controller + Gripper (Final)

사용법:
  터미널 1: ros2 launch piper start_single_piper.launch.py
  터미널 2: ros2 launch piper_moveit demo.launch.py
  터미널 3: python3 move_piper_final.py

실행 환경:
  cd ~/piper_ws && source install/setup.bash
  cd /mnt/c/Users/Ryuminseo/source/repos/DLIP/Final/Robot/Python
  python3 move_piper_final.py
"""

import rclpy
import threading
import math
import time
import sys
import select
import tty
import termios
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from pymoveit2 import MoveIt2
from tf2_ros import Buffer, TransformListener
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool


# ── 로봇 설정 ─────────────────────────────────────────────────────────
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
BASE_LINK   = "base_link"
EEF_LINK    = "link6"
GROUP_NAME  = "Manipulator"

HOME_JOINTS = [0.0, math.radians(20), math.radians(-30), 0.0, math.radians(10), 0.0]

MAX_REACH_MM = 627.0
MIN_Z_MM     = -10.0

PLANNERS = [
    "RRTConnectkConfigDefault",
    "BiTRRTkConfigDefault",
    "LBKPIECEkConfigDefault",
]

BRIDGE_PITCHES = [90.0, 60.0, 30.0]
# ──────────────────────────────────────────────────────────────────────

# ── 그리퍼 설정 ───────────────────────────────────────────────────────
GRIPPER_OPEN  = 0.05
GRIPPER_CLOSE = 0.005
# ──────────────────────────────────────────────────────────────────────


def euler_to_quat_xyzw(roll_deg: float, pitch_deg: float, yaw_deg: float) -> list:
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return [
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ]


def quat_to_euler_deg(qx, qy, qz, qw):
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def get_current_pose_mm(tf_buffer):
    try:
        t = tf_buffer.lookup_transform(BASE_LINK, EEF_LINK, rclpy.time.Time())
        tr = t.transform.translation
        ro = t.transform.rotation
        x = tr.x * 1000.0
        y = tr.y * 1000.0
        z = tr.z * 1000.0
        roll, pitch, yaw = quat_to_euler_deg(ro.x, ro.y, ro.z, ro.w)
        return x, y, z, roll, pitch, yaw
    except Exception:
        return None


def print_current_pose(tf_buffer, label="현재 위치"):
    pose = get_current_pose_mm(tf_buffer)
    if pose:
        x, y, z, roll, pitch, yaw = pose
        print(f"  [{label}]  X={x:7.1f}mm  Y={y:7.1f}mm  Z={z:7.1f}mm"
              f"   R={roll:6.1f}°  P={pitch:6.1f}°  Yaw={yaw:6.1f}°")
    else:
        print(f"  [{label}]  TF 데이터 없음")


def ask_float(prompt, default=None):
    while True:
        try:
            raw = input(prompt).strip()
            if raw == "" and default is not None:
                return default
            return float(raw)
        except ValueError:
            print("  ※ 숫자를 입력하세요.")


def is_reachable(x_mm, y_mm, z_mm):
    dist = math.sqrt(x_mm**2 + y_mm**2 + z_mm**2)
    if dist > MAX_REACH_MM:
        return False, f"반경 {dist:.1f}mm > 최대 {MAX_REACH_MM}mm"
    if z_mm < MIN_Z_MM:
        return False, f"Z={z_mm:.1f}mm < 하한 {MIN_Z_MM}mm"
    return True, ""


def wait_with_estop(moveit2):
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    done = threading.Event()
    result = [False]

    def _wait():
        result[0] = bool(moveit2.wait_until_executed())
        done.set()

    wt = threading.Thread(target=_wait, daemon=True)
    emergency_stopped = False

    try:
        tty.setcbreak(fd)
        while select.select([sys.stdin], [], [], 0.0)[0]:
            sys.stdin.read(1)
        wt.start()
        while not done.is_set():
            if select.select([sys.stdin], [], [], 0.05)[0]:
                ch = sys.stdin.read(1)
                if ch.lower() == 's':
                    print("\n\n  !! 비상 정지 !! 정지 명령 전송 중...")
                    try:
                        moveit2.cancel_execution()
                    except Exception as e:
                        print(f"  (cancel_execution 오류: {e})")
                    emergency_stopped = True
                    done.wait(timeout=3.0)
                    break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        wt.join(timeout=1.0)

    return result[0], emergency_stopped


def _try_move(moveit2, position, quat, planner):
    moveit2.planner_id = planner
    moveit2.move_to_pose(position=position, quat_xyzw=quat)
    return wait_with_estop(moveit2)


def move_direct(moveit2, position, roll, pitch, yaw):
    quat = euler_to_quat_xyzw(roll, pitch, yaw)
    for planner in PLANNERS:
        short = planner.replace("kConfigDefault", "")
        print(f"  [직접] 플래너={short} ... ('s'=비상정지)", end=" ", flush=True)
        success, stopped = _try_move(moveit2, position, quat, planner)
        if stopped:
            return False, True
        if success:
            print("✓")
            return True, False
        print("✗")
    return False, False


def move_with_bridge(moveit2, tf_buffer, position, roll, pitch, yaw):
    for br_pitch in BRIDGE_PITCHES:
        if abs(br_pitch - pitch) < 5.0:
            continue
        print(f"\n  [브릿지] Pitch={br_pitch:.0f}° 경유 시도...")
        success, stopped = move_direct(moveit2, position, roll, br_pitch, yaw)
        if stopped:
            return False, True
        if not success:
            print(f"  [브릿지] Pitch={br_pitch:.0f}° 실패, 다음으로...")
            continue
        time.sleep(0.3)
        print_current_pose(tf_buffer, f"브릿지 도달(P={br_pitch:.0f}°)")
        print(f"  [브릿지→목표] 목표 Pitch={pitch:.0f}° 재시도...")
        success2, stopped2 = move_direct(moveit2, position, roll, pitch, yaw)
        if stopped2:
            return False, True
        if success2:
            return True, False
        print(f"  [브릿지→목표] 실패. 홈 복귀 후 다음 경유점 시도...")
        go_home(moveit2, tf_buffer)
    return False, False


def go_home(moveit2, tf_buffer):
    print("  [홈 복귀 중... ('s'=비상정지)]")
    moveit2.planner_id = PLANNERS[0]
    moveit2.move_to_configuration(joint_positions=HOME_JOINTS)
    _, stopped = wait_with_estop(moveit2)
    if stopped:
        print("  !! 홈 복귀 중 비상 정지 — 로봇 상태를 수동으로 확인하세요.")
        time.sleep(0.3)
        print_current_pose(tf_buffer, "정지 위치")
        return False
    print("  홈 포지션 완료.")
    time.sleep(0.3)
    print_current_pose(tf_buffer, "홈 위치")
    return True


def main():
    rclpy.init()
    node = Node("move_piper_final")
    cb   = ReentrantCallbackGroup()

    moveit2 = MoveIt2(
        node=node,
        joint_names=JOINT_NAMES,
        base_link_name=BASE_LINK,
        end_effector_name=EEF_LINK,
        group_name=GROUP_NAME,
        callback_group=cb,
    )

    moveit2.allowed_planning_time        = 15.0
    moveit2.num_planning_attempts        = 50
    moveit2.max_velocity_scaling_factor     = 0.3
    moveit2.max_acceleration_scaling_factor = 0.3

    executor = MultiThreadedExecutor(2)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    tf_buffer   = Buffer()
    tf_listener = TransformListener(tf_buffer, node)  # noqa: F841

    enable_pub  = node.create_publisher(Bool, '/enable_flag', 10)
    gripper_pub = node.create_publisher(JointState, '/joint_states', 10)

    def read_joint_positions():
        js = moveit2.joint_state
        if js is None or len(js.position) < len(js.name):
            return None
        pos = [0.0] * 6
        for name, val in zip(js.name, js.position):
            if name in JOINT_NAMES:
                pos[JOINT_NAMES.index(name)] = val
        return pos

    def set_gripper(position, speed=0.3):
        arm_pos = list(HOME_JOINTS)
        try:
            read = read_joint_positions()
            if read is not None:
                arm_pos = read
        except Exception:
            pass

        msg = JointState()
        msg.header.frame_id = 'piper_single'
        msg.name     = ['joint1', 'joint2', 'joint3',
                        'joint4', 'joint5', 'joint6',
                        'joint7', 'joint8']
        msg.position = arm_pos + [float(position), 0.0]
        msg.velocity = [speed] * 8
        msg.effort   = [0.0] * 7 + [0.2]

        for _ in range(5):
            msg.header.stamp = node.get_clock().now().to_msg()
            gripper_pub.publish(msg)
            time.sleep(0.05)

        state = "열림" if position >= GRIPPER_OPEN else ("닫힘" if position <= GRIPPER_CLOSE else f"{position:.3f}m")
        print(f"  [그리퍼] {state}")

    def ask_gripper():
        print(f"\n  ─ 그리퍼 ──────────────────────────────────────────")
        print(f"  'o' = 열기  /  'c' = 닫기  /  숫자 = 직접 입력  /  Enter = 건너뜀")
        print( "  ────────────────────────────────────────────────────")
        raw = input("  그리퍼: ").strip().lower()

        if raw == 'o':
            set_gripper(GRIPPER_OPEN)
            time.sleep(1.0)
        elif raw == 'c':
            set_gripper(GRIPPER_CLOSE)
            time.sleep(1.0)
        elif raw != '':
            try:
                val = float(raw)
                val = max(GRIPPER_CLOSE, min(GRIPPER_OPEN, val))
                set_gripper(val)
                time.sleep(1.0)
            except ValueError:
                print("  ※ 유효하지 않은 값 — 그리퍼 변경 없음")

    print("\n" + "=" * 60)
    print("   PiPER Final Controller")
    print("   이동 중 's' → 비상정지 / X 입력에 'r' → 홈 복귀 / 'j' → 관절 직접 제어 / 'q' → 종료")
    print("=" * 60)
    print("  MoveIt2 서버 연결 대기 중...")
    time.sleep(2.0)

    print("[초기화] 로봇 활성화...")
    enable_msg = Bool()
    enable_msg.data = True
    enable_pub.publish(enable_msg)
    time.sleep(1.0)

    print("[초기화] 홈 포지션으로 이동합니다...")
    go_home(moveit2, tf_buffer)
    print("[초기화] 그리퍼 닫기...")
    set_gripper(GRIPPER_CLOSE)
    time.sleep(1.0)

    while True:
        print("\n" + "-" * 60)
        print(f"  ※ 작업 반경 최대 {MAX_REACH_MM:.0f}mm, Z 하한 {MIN_Z_MM:.0f}mm")
        print("  ※ 이동 중 's' → 비상정지  /  'r' → 홈 복귀  /  'j' → 관절 직접  /  'q' → 종료")
        print_current_pose(tf_buffer)
        print("-" * 60)

        x_raw = input("  X (mm): ").strip().lower()

        if x_raw == "q":
            print("\n[종료] 모든 관절을 0으로 이동 중...")
            moveit2.planner_id = PLANNERS[0]
            moveit2.move_to_configuration(joint_positions=[0.0] * 6)
            wait_with_estop(moveit2)
            set_gripper(GRIPPER_CLOSE)
            print("종료합니다.")
            break

        if x_raw == "r":
            go_home(moveit2, tf_buffer)
            ask_gripper()
            continue

        if x_raw == "j":
            current = read_joint_positions()
            if current is None:
                print("  ※ 관절 상태를 읽을 수 없습니다.")
                continue
            print(f"  [현재 관절각(deg)] " +
                  "  ".join(f"J{i+1}={math.degrees(v):6.1f}°" for i, v in enumerate(current)))
            print("  변경할 관절 번호(1-6)와 추가 각도(deg)를 입력하세요.")
            print("  예) 6 90  →  Joint6에 +90° 추가 / 빈 칸 Enter = 변경 없음")
            try:
                raw = input("  관절번호 델타(deg): ").strip()
                if not raw:
                    continue
                parts = raw.split()
                jidx = int(parts[0]) - 1
                delta = float(parts[1])
                if not (0 <= jidx <= 5):
                    print("  ※ 관절 번호는 1~6")
                    continue
                current[jidx] += math.radians(delta)
                print(f"  → J{jidx+1}에 {delta:+.1f}° 적용: {math.degrees(current[jidx]):.1f}°")
            except (ValueError, IndexError):
                print("  ※ 입력 형식 오류. 예: 6 90")
                continue
            moveit2.planner_id = PLANNERS[0]
            moveit2.move_to_configuration(joint_positions=current)
            success, stopped = wait_with_estop(moveit2)
            if stopped:
                print("  !! 비상 정지")
            elif success:
                print("  ✓ 관절 이동 완료")
                time.sleep(0.3)
                print_current_pose(tf_buffer, "도달 위치")
                ask_gripper()
            else:
                print("  ✗ 관절 이동 실패")
            continue

        if x_raw == "":
            continue

        try:
            x_mm  = float(x_raw)
            y_mm  = ask_float("  Y (mm): ")
            z_mm  = ask_float("  Z (mm): ")
            roll  = ask_float("  Roll  (deg) [기본 0]: ", default=0.0)
            pitch = ask_float("  Pitch (deg) [기본 0]: ", default=0.0)
            yaw   = ask_float("  Yaw   (deg) [기본 0]: ", default=0.0)
        except (ValueError, EOFError):
            print("  ※ 입력 오류. 다시 시도하세요.")
            continue

        ok, reason = is_reachable(x_mm, y_mm, z_mm)
        if not ok:
            print(f"  ✗ 목표 거부: {reason}")
            continue

        position = [x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0]
        print(f"\n  → 이동 시작: X={x_mm}mm  Y={y_mm}mm  Z={z_mm}mm  "
              f"R={roll}°  P={pitch}°  Yaw={yaw}°")

        success, emergency_stopped = move_direct(moveit2, position, roll, pitch, yaw)

        if not success and not emergency_stopped:
            print("  [직접 경로 실패] 브릿지 경유 전략 시도...")
            success, emergency_stopped = move_with_bridge(
                moveit2, tf_buffer, position, roll, pitch, yaw)

        if emergency_stopped:
            print("\n  !! 비상 정지 완료.")
            time.sleep(0.3)
            print_current_pose(tf_buffer, "정지 위치")
            input("  [Enter] 를 누르면 홈으로 복귀합니다...")
            go_home(moveit2, tf_buffer)
            ask_gripper()

        elif success:
            print("  ✓ 목표 위치 도달!")
            time.sleep(0.3)
            print_current_pose(tf_buffer, "도달 위치")
            ask_gripper()

        else:
            print("  ✗ 이동 실패 — 좌표 또는 방향을 수정하세요.")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
