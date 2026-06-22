#!/usr/bin/env python3
"""
Final_Robot.py  (WSL 실행)

Windows Final_Camera.py로부터 목표 수신 → 전체 시퀀스 자율 실행

시퀀스:
  그리퍼 열기 → APPROACH → MOVE → 그리퍼 닫기(GRASP)
  → HOME → PLACE_MOVE → [WSL 터미널 Space/Enter] 그리퍼 열기 → HOME → done

실행:
  터미널 1: ros2 launch piper start_single_piper.launch.py
  터미널 2: ros2 launch piper_moveit demo.launch.py
  터미널 3:
    cd ~/piper_ws && source install/setup.bash
    cd /mnt/c/Users/Ryuminseo/source/repos/DLIP/Final/Main
    python3 Final_Robot.py

이동 중:
  s : 비상 정지
PLACE 위치 도달 후:
  Space 또는 Enter : 그리퍼 열기 (물체 내려놓기)
"""

import json
import math
import select
import socket
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from pymoveit2 import MoveIt2
from tf2_ros import Buffer, TransformListener
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool


# ── 설정 ─────────────────────────────────────────────────────────────────────
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
BASE_LINK   = "base_link"
EEF_LINK    = "link6"
GROUP_NAME  = "Manipulator"

HOME_JOINTS = [0.0, math.radians(20), math.radians(-30), 0.0, math.radians(10), 0.0]

MAX_REACH_MM = 627.0
MIN_Z_MM     = 70.0

PLANNERS = [
    "RRTConnectkConfigDefault",
    "BiTRRTkConfigDefault",
    "LBKPIECEkConfigDefault",
]

GRIPPER_OPEN  = 0.05
GRIPPER_CLOSE = 0.01

LISTEN_HOST = "0.0.0.0"
LISTEN_PORT = 7777

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  ★ PLACE Yaw 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PLACE_YAW_NORMAL : float =  90.0
PLACE_YAW_FLIPPED: float = -90.0
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class RobotCtx:
    node:        Node
    moveit2:     MoveIt2
    tf_buffer:   Buffer
    gripper_pub: object


# ── 수학 유틸 ─────────────────────────────────────────────────────────────────
def euler_to_quat_xyzw(roll_deg: float, pitch_deg: float, yaw_deg: float) -> list:
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    cr, sr = math.cos(r/2), math.sin(r/2)
    cp, sp = math.cos(p/2), math.sin(p/2)
    cy, sy = math.cos(y/2), math.sin(y/2)
    return [sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy,
            cr*cp*cy + sr*sp*sy]


def build_place_target(place: dict, blade_flipped: bool) -> dict:
    adjusted = dict(place)
    if blade_flipped:
        adjusted["yaw"] = PLACE_YAW_FLIPPED
        print(f"  [PLACE] blade_flipped=True  → Yaw = {PLACE_YAW_FLIPPED:+.1f}°")
    else:
        adjusted["yaw"] = PLACE_YAW_NORMAL
        print(f"  [PLACE] blade_flipped=False → Yaw = {PLACE_YAW_NORMAL:+.1f}°")
    return adjusted


# ── 비상 정지 포함 이동 대기 ──────────────────────────────────────────────────
def wait_with_estop(moveit2: MoveIt2) -> Tuple[bool, bool]:
    fd           = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    done         = threading.Event()
    result       = [False]

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
                    print("\n  !! 비상 정지 !!")
                    try:
                        moveit2.cancel_execution()
                    except Exception as e:
                        print(f"  (cancel 오류: {e})")
                    emergency_stopped = True
                    done.wait(timeout=3.0)
                    break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        wt.join(timeout=1.0)
    return result[0], emergency_stopped


def _try_move(moveit2: MoveIt2, position: list,
              quat: list, planner: str) -> Tuple[bool, bool]:
    moveit2.planner_id = planner
    moveit2.move_to_pose(position=position, quat_xyzw=quat)
    return wait_with_estop(moveit2)


def move_direct(moveit2: MoveIt2, position: list,
                roll: float, pitch: float, yaw: float) -> Tuple[bool, bool]:
    quat = euler_to_quat_xyzw(roll, pitch, yaw)
    for planner in PLANNERS:
        short = planner.replace("kConfigDefault", "")
        print(f"  [직접] {short} ... ('s'=비상정지)", end=" ", flush=True)
        success, stopped = _try_move(moveit2, position, quat, planner)
        if stopped:
            return False, True
        if success:
            print("✓")
            return True, False
        print("✗")
    return False, False


def go_home_raw(moveit2: MoveIt2, tf_buffer: Buffer) -> bool:
    print("  [홈 복귀...]")
    moveit2.planner_id = PLANNERS[0]
    moveit2.move_to_configuration(joint_positions=HOME_JOINTS)
    _, stopped = wait_with_estop(moveit2)
    if stopped:
        print("  !! 홈 복귀 중 비상 정지")
        return False
    print("  홈 완료.")
    return True


def read_joint_positions(moveit2: MoveIt2):
    js = moveit2.joint_state
    if js is None or len(js.position) < len(js.name):
        return None
    pos = [0.0] * 6
    for name, val in zip(js.name, js.position):
        if name in JOINT_NAMES:
            pos[JOINT_NAMES.index(name)] = val
    return pos


def set_gripper(ctx: RobotCtx, position: float, speed: float = 0.3):
    arm_pos = list(HOME_JOINTS)
    for _ in range(10):
        try:
            read = read_joint_positions(ctx.moveit2)
            if read is not None:
                arm_pos = read
                break
        except Exception:
            pass
        time.sleep(0.05)
    msg = JointState()
    msg.header.frame_id = 'piper_single'
    msg.name     = ['joint1', 'joint2', 'joint3',
                    'joint4', 'joint5', 'joint6',
                    'joint7', 'joint8']
    msg.position = arm_pos + [float(position), 0.0]
    msg.velocity = [speed] * 8
    msg.effort   = [0.0] * 7 + [0.2]
    for _ in range(10):
        msg.header.stamp = ctx.node.get_clock().now().to_msg()
        ctx.gripper_pub.publish(msg)
        time.sleep(0.1)
    print(f"  [그리퍼] {'열림' if position >= GRIPPER_OPEN else '닫힘'}")


def move_joint_space(moveit2: MoveIt2, joints: list, label: str = "JOINT") -> Tuple[bool, bool]:
    print(f"\n[{label}] " + "  ".join(f"J{i+1}={math.degrees(v):.1f}°" for i, v in enumerate(joints)))
    moveit2.planner_id = PLANNERS[0]
    moveit2.move_to_configuration(joint_positions=joints)
    return wait_with_estop(moveit2)


def action_move(ctx: RobotCtx, target: dict,
                label: str = "MOVE") -> Tuple[bool, bool]:
    pos = [target["x_mm"]/1000.0,
           target["y_mm"]/1000.0,
           target["z_mm"]/1000.0]
    print(f"\n[{label}] X={target['x_mm']:.1f} Y={target['y_mm']:.1f} "
          f"Z={target['z_mm']:.1f}mm  "
          f"R={target['roll']:.0f} P={target['pitch']:.0f} Yaw={target['yaw']:.1f}°")
    return move_direct(ctx.moveit2, pos,
                       target["roll"], target["pitch"], target["yaw"])


# ── 키 입력 대기 (WSL 터미널) ────────────────────────────────────────────────
def wait_for_keypress(prompt: str, tag: str) -> bool:
    """Space 또는 Enter 입력 대기. True=계속, False=비상정지"""
    print(f"\n[{tag}] ★ {prompt}  /  s → 비상 정지 ★")
    fd           = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        termios.tcflush(fd, termios.TCIFLUSH)
        while True:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                ch = sys.stdin.read(1)
                if ch in (' ', '\r', '\n'):
                    print(f"[{tag}] 입력 감지")
                    return True
                if ch.lower() == 's':
                    print(f"\n  !! 비상 정지 ({tag}) !!")
                    return False
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def wait_for_grasp() -> bool:
    return wait_for_keypress("Space 또는 Enter → 그리퍼 닫기(GRASP)", "GRASP_WAIT")


def wait_for_release() -> bool:
    return wait_for_keypress("Space 또는 Enter → 그리퍼 열기(RELEASE)", "PLACE_WAIT")


class _Disconnect(BaseException):
    pass


# ── 클라이언트 처리 ───────────────────────────────────────────────────────────
def handle_client(conn: socket.socket, ctx: RobotCtx):
    f = conn.makefile("r")

    def send(msg: dict):
        try:
            conn.sendall((json.dumps(msg) + "\n").encode())
        except Exception:
            pass

    send({"status": "ready"})
    print("[SOCKET] Windows 연결됨\n")

    try:
        while True:
            try:
                line = f.readline()
            except Exception:
                break
            if not line:
                break
            try:
                cmd = json.loads(line.strip())
            except json.JSONDecodeError:
                continue

            action = cmd.get("cmd")

            # ── 홈 복귀 ───────────────────────────────────────────────────
            if action == "home":
                print("[CMD] 홈 복귀")
                go_home_raw(ctx.moveit2, ctx.tf_buffer)
                set_gripper(ctx, GRIPPER_CLOSE)
                send({"status": "ready"})

            # ── 전체 시퀀스 (자율) ────────────────────────────────────────
            elif action == "execute":
                approach      = cmd["approach"]
                target        = cmd["target"]
                place         = cmd.get("place")
                blade_flipped = target.get("blade_flipped", False)

                print(f"\n{'='*50}")
                print(f"[EXECUTE] {target.get('cls_name', '?')}  blade_flipped={blade_flipped}")
                print(f"{'='*50}")

                # set_gripper publishes all joint positions, so keep it separate
                # from MoveIt execution to avoid fighting the active trajectory.
                print("\n[APPROACH] 그리퍼 열기...")
                set_gripper(ctx, GRIPPER_OPEN)
                time.sleep(0.5)
                print("\n[APPROACH] 이동...")
                success, estop = action_move(ctx, approach, label="APPROACH")
                if estop or not success:
                    go_home_raw(ctx.moveit2, ctx.tf_buffer)
                    set_gripper(ctx, GRIPPER_CLOSE)
                    send({"status": "estop" if estop else "failed",
                          "msg": "approach failed"})
                    continue

                print("[APPROACH] 완료 — 2초 대기...")
                time.sleep(2.0)

                # MOVE
                print("\n[MOVE]")
                success, estop = action_move(ctx, target, label="MOVE")
                if estop or not success:
                    go_home_raw(ctx.moveit2, ctx.tf_buffer)
                    set_gripper(ctx, GRIPPER_CLOSE)
                    send({"status": "estop" if estop else "failed",
                          "msg": "move failed"})
                    continue

                print("[MOVE] 완료 — 2초 후 GRASP...")
                time.sleep(2.0)

                # GRASP
                print("\n[GRASP]")
                set_gripper(ctx, GRIPPER_CLOSE)
                time.sleep(1.0)

                # HOME
                print("\n[HOME]")
                go_home_raw(ctx.moveit2, ctx.tf_buffer)

                # PLACE_MOVE
                if place:
                    print("\n[PLACE_MOVE] 배치 위치로 이동...")
                    if "joints" in place:
                        success, estop = move_joint_space(
                            ctx.moveit2, place["joints"], label="PLACE_MOVE")
                    else:
                        place_adjusted = build_place_target(place, blade_flipped)
                        success, estop = action_move(ctx, place_adjusted, label="PLACE_MOVE")

                    if estop or not success:
                        set_gripper(ctx, GRIPPER_OPEN)
                        time.sleep(0.5)
                        go_home_raw(ctx.moveit2, ctx.tf_buffer)
                        set_gripper(ctx, GRIPPER_CLOSE)
                        send({"status": "estop" if estop else "failed",
                              "msg": "place_move failed"})
                        continue

                    print("[PLACE_MOVE] 완료 — Space/Enter 입력 대기...")
                    if not wait_for_release():
                        send({"status": "estop",
                              "msg": "release cancelled"})
                        continue

                    set_gripper(ctx, GRIPPER_OPEN)
                    time.sleep(0.5)
                    go_home_raw(ctx.moveit2, ctx.tf_buffer)
                    set_gripper(ctx, GRIPPER_CLOSE)
                else:
                    go_home_raw(ctx.moveit2, ctx.tf_buffer)
                    set_gripper(ctx, GRIPPER_CLOSE)

                print("\n[DONE] 시퀀스 완료\n")
                send({"status": "done"})

            # ── 종료 ──────────────────────────────────────────────────────
            elif action == "quit":
                print("[CMD] 종료")
                set_gripper(ctx, GRIPPER_CLOSE)
                ctx.moveit2.planner_id = PLANNERS[0]
                ctx.moveit2.move_to_configuration(
                    joint_positions=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                wait_with_estop(ctx.moveit2)
                break

    except _Disconnect:
        pass
    finally:
        f.close()
        conn.close()
        print("[SOCKET] 연결 종료")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Final_Robot.py  (WSL)")
    print(f"  PLACE Yaw: 정방향={PLACE_YAW_NORMAL:+.1f}°  반전={PLACE_YAW_FLIPPED:+.1f}°")
    print("=" * 60)

    rclpy.init()
    node = Node("final_robot")
    cb   = ReentrantCallbackGroup()

    moveit2 = MoveIt2(
        node=node,
        joint_names=JOINT_NAMES,
        base_link_name=BASE_LINK,
        end_effector_name=EEF_LINK,
        group_name=GROUP_NAME,
        callback_group=cb,
    )
    moveit2.allowed_planning_time           = 15.0
    moveit2.num_planning_attempts           = 50
    moveit2.max_velocity_scaling_factor     = 0.3
    moveit2.max_acceleration_scaling_factor = 0.3

    executor = MultiThreadedExecutor(2)
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()

    tf_buffer   = Buffer()
    tf_listener = TransformListener(tf_buffer, node)  # noqa: F841

    enable_pub  = node.create_publisher(Bool,       '/enable_flag',  10)
    gripper_pub = node.create_publisher(JointState, '/joint_states', 10)

    ctx = RobotCtx(node=node, moveit2=moveit2,
                   tf_buffer=tf_buffer, gripper_pub=gripper_pub)

    print("[Initializing] MoveIt2 연결 중...")
    time.sleep(2.0)

    enable_msg = Bool()
    enable_msg.data = True
    enable_pub.publish(enable_msg)
    time.sleep(1.0)

    print("[Initializing] 홈 포지션으로 이동...")
    go_home_raw(ctx.moveit2, ctx.tf_buffer)
    set_gripper(ctx, GRIPPER_CLOSE)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((LISTEN_HOST, LISTEN_PORT))
    server.listen(1)
    print(f"\n[SOCKET] TCP 서버 대기 중 (0.0.0.0:{LISTEN_PORT})")
    print("         → Windows에서 Final_Camera.py를 실행하세요.\n")

    try:
        while True:
            conn, addr = server.accept()
            print(f"[SOCKET] 연결: {addr}")
            handle_client(conn, ctx)
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        node.destroy_node()
        rclpy.shutdown()
        print("[종료]")


if __name__ == "__main__":
    main()
