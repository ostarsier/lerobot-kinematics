import os
import mujoco
import mujoco.viewer
import numpy as np
import time
from IK_SO100_box import rtb_Kinematics
from pynput import keyboard
import threading

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
JOINT_INCREMENT = 0.01  # 可以根据需要调整

# 定义关节限幅：
qlimit = [[-2.1, -3.0, -0.1, -2.0, -3.0, -0.1], 
          [2.1,   0.2,   3.0, 1.8,  3.0, 1]]

# 初始化目标关节位置
init_qpos = np.array([0, -3.14, 3.14, 0, 0, -0.157])
# init_qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
target_qpos = init_qpos.copy() # mjdata.qpos[qpos_indices].copy()
init_pose = rtb_kinematics.rtb_forward_kinematics(np.zeros(6))

# 线程安全锁
lock = threading.Lock()

# 定义键映射
key_to_joint_increase = {
    '1': 0,  # Rotation
    '2': 1,  # Pitch
    '3': 2,  # Elbow
    '4': 3,  # Wrist_Pitch
    '5': 4,  # Wrist_Roll
    '6': 5,  # Jaw
}

key_to_joint_decrease = {
    'q': 0,  # Rotation
    'w': 1,  # Pitch
    'e': 2,  # Elbow
    'r': 3,  # Wrist_Pitch
    't': 4,  # Wrist_Roll
    'y': 5,  # Jaw
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
                global target_qpos
                target_qpos = init_qpos.copy()

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
        
try:
    with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
        
        start = time.time()
        t = 0
        while viewer.is_running() and time.time() - start < 1000:
            step_start = time.time()
            
            # 更新 target_qpos 基于当前按下的键
            with lock:
                for k, direction in keys_pressed.items():
                    if k in key_to_joint_increase:
                        joint_idx = key_to_joint_increase[k]
                        if target_qpos[joint_idx] < qlimit[1][joint_idx]:
                            target_qpos[joint_idx] += JOINT_INCREMENT * direction
                        
                    elif k in key_to_joint_decrease:
                        joint_idx = key_to_joint_decrease[k]
                        if target_qpos[joint_idx] > qlimit[0][joint_idx]:
                            target_qpos[joint_idx] += JOINT_INCREMENT * direction  # direction 为 -1

            # 前向和逆向运动学
            # print(f'{target_qpos=}')
            position = rtb_kinematics.rtb_forward_kinematics(target_qpos[:5])
            print("tg_qpos:", [f"{x:.3f}" for x in target_qpos])
            # print("fk_gpos:", [f"{x:.3f}" for x in position])
            qpos_inv = rtb_kinematics.rtb_inverse_kinematics(target_qpos[:5], position)
            
            # 使用逆解保护一下
            if qpos_inv[0] != -1.0 and qpos_inv[1] != -1.0 and qpos_inv[2] != -1.0 and qpos_inv[3] != -1.0:
                target_qpos = np.concatenate((qpos_inv, target_qpos[5:]))
                mjdata.qpos[qpos_indices] = target_qpos
                
                # 步进模拟
                mjdata.qpos[qpos_indices] = np.concatenate((qpos_inv,target_qpos[5:]))
                mujoco.mj_step(mjmodel, mjdata)

                # 更新查看器选项（例如，每秒切换一次接触点显示）
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(mjdata.time % 2)

                # 同步查看器
                viewer.sync()
                position = rtb_kinematics.rtb_forward_kinematics(mjdata.qpos[qpos_indices]) 
                print("fb_qpos:", [f"{x:.3f}" for x in mjdata.qpos[qpos_indices]])
                print()
                
                target_gpos_last = target_qpos.copy() # 保存备份
            else:
                target_qpos = target_gpos_last.copy()
            
            # 时间管理以维持模拟时间步
            time_until_next_step = mjmodel.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
except KeyboardInterrupt:
    print("用户中断了模拟。")
finally:
    listener.stop()
