import os
import mujoco
import mujoco.viewer
import numpy as np
import time
from IK_SO100_box_gpos import rtb_Kinematics
from pynput import keyboard
import threading
from collections import deque

np.set_printoptions(linewidth=200)
# 设置 MuJoCo 渲染后端
os.environ["MUJOCO_GL"] = "egl"

# 定义关节名称
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

# XML 模型的绝对路径
xml_path = "/home/boxjod/lerobot/lerobot/sim_env/scene.xml"
mjmodel = mujoco.MjModel.from_xml_path(xml_path)
qpos_indices = np.array([mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in JOINT_NAMES])
mjdata = mujoco.MjData(mjmodel)

# 初始化运动学
rtb_kinematics = rtb_Kinematics()

# 定义关节控制增量（弧度）
JOINT_INCREMENT = 0.001  # 可以根据需要调整
POSITION_INSERMENT = 0.0008

# 定义关节限幅：
qlimit = [[-2.1, -3.14, -0.1, -2.0, -3.1, -0.1], 
          [2.1,   0.2,   3.14, 1.8,  3.1, 1.0]]
glimit = [[0.000, -0.4,  0.046, -3.1, -1.5, -1.5], 
          [0.430,  0.4,  0.23,  3.1,  1.5,  1.5]]

# 初始化目标关节位置
# init_qpos = np.array([0.0, -3.14, 3.14, 0.0, 0.0, -0.157])
init_qpos = np.array([0.0, -3.14, 3.14, 0.0, -1.57, -0.157])
# init_qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
target_qpos = init_qpos.copy() # mjdata.qpos[qpos_indices].copy()
init_gpos = rtb_kinematics.rtb_forward_kinematics(init_qpos[1:5])
target_gpos = init_gpos.copy() 

# 线程安全锁
lock = threading.Lock()

# 定义键映射
key_to_joint_increase = {
    'w': 0,  # 前进
    'a': 1,  # 右
    'r': 2,  # 上
    'e': 3,  # roll+
    't': 4,  # pitch+
    'z': 5,  # gripper+
}

key_to_joint_decrease = {
    's': 0,  # 后退
    'd': 1,  # 左
    'f': 2,  # 下
    'q': 3,  # roll-
    'g': 4,  # pitch-
    'c': 5,  # gripper-
}

# 字典用于跟踪当前按下的键及其增减方向
keys_pressed = {}

def on_press(key):
    try:
        k = key.char.lower()  # 转为小写以处理大写输入
        if k in key_to_joint_increase:
            with lock:
                keys_pressed[k] = 1  # 增加方向
        elif k in key_to_joint_decrease:
            with lock:
                keys_pressed[k] = -1  # 减少方向
        elif k == "0":
            with lock:
                global target_qpos, target_gpos
                target_qpos = init_qpos.copy()
                target_gpos = init_gpos.copy()
        print(f'{key}')

    except AttributeError:
        pass  # 处理特殊键（如果需要）

def on_release(key):
    try:
        k = key.char.lower()
        if k in keys_pressed:
            with lock:
                del keys_pressed[k]
    except AttributeError:
        pass  # 处理特殊键（如果需要）

# 启动键盘监听器在一个单独的线程
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

target_gpos_last = init_gpos.copy()

def calc_target_distance(_target_gpos):
    x, y, z, r, p, y = _target_gpos
    distance = np.sqrt(x*x + y*y + z*z)
    # print(f'{distance=}')
    return distance

try:
    with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
        
        start = time.time()
        t = 0
        while viewer.is_running() and time.time() - start < 1000:
            step_start = time.time()

            with lock:
                for k, direction in keys_pressed.items():
                    if k in key_to_joint_increase:
                        position_idx = key_to_joint_increase[k]
                        if position_idx == 1 or position_idx == 5: 
                            position_idx = 0 if position_idx == 1 else 5
                            if (target_qpos[position_idx]) < qlimit[1][position_idx] - JOINT_INCREMENT * direction:
                                target_qpos[position_idx] += JOINT_INCREMENT * direction
                        elif position_idx == 4 or position_idx == 3:
                            if target_gpos[position_idx] <= glimit[1][position_idx]  :#and calc_target_distance(target_gpos) < 0.36:
                                target_gpos[position_idx] += POSITION_INSERMENT * direction*4
                        else:
                            if target_gpos[position_idx] <= glimit[1][position_idx]  :#and calc_target_distance(target_gpos) < 0.36:
                                target_gpos[position_idx] += POSITION_INSERMENT * direction
                        
                    elif k in key_to_joint_decrease:
                        position_idx = key_to_joint_decrease[k]
                        if position_idx == 1 or position_idx == 5:
                            position_idx = 0 if position_idx == 1 else 5
                            if (target_qpos[position_idx]) > qlimit[0][position_idx] - JOINT_INCREMENT * direction:
                                target_qpos[position_idx] += JOINT_INCREMENT * direction
                                # target_gpos = rtb_kinematics.rtb_forward_kinematics(target_qpos[1:5])
                        elif position_idx == 4 or position_idx == 3:
                            if target_gpos[position_idx] <= glimit[1][position_idx]  :#and calc_target_distance(target_gpos) < 0.36:
                                target_gpos[position_idx] += POSITION_INSERMENT * direction*4
                        else:
                            if target_gpos[position_idx] >= glimit[0][position_idx] :# and calc_target_distance(target_gpos) > 0.15:
                                target_gpos[position_idx] += POSITION_INSERMENT * direction  # direction 为 -1
                position = target_gpos
            print(f'{target_gpos=}')
            qpos_inv = rtb_kinematics.rtb_inverse_kinematics(mjdata.qpos[qpos_indices][1:5], position)
            if qpos_inv[0] != -1.0 and qpos_inv[1] != -1.0 and qpos_inv[2] != -1.0 and qpos_inv[3] != -1.0:
                target_qpos = np.concatenate((target_qpos[0:1], qpos_inv, target_qpos[5:]))
                mjdata.qpos[qpos_indices] = target_qpos
                print(f'{target_qpos=}')
                # 步进模拟
                mujoco.mj_step(mjmodel, mjdata)
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(mjdata.time % 2)
                viewer.sync()
                
                target_gpos_last = position.copy() # 保存备份
            else:
                target_gpos = target_gpos_last.copy()
                position = target_gpos
            
            # 时间管理以维持模拟时间步
            time_until_next_step = mjmodel.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
except KeyboardInterrupt:
    print("用户中断了模拟。")
finally:
    listener.stop()
