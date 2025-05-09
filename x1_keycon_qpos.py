# code by AI Assistant 2024.05.30 基于lerobot_keycon_qpos.py修改，适用于x1机器人模型

import os
import mujoco
import mujoco.viewer
import numpy as np
import time
from pynput import keyboard
import threading

# 设置NumPy打印选项，使数组输出更宽，便于查看
np.set_printoptions(linewidth=200)

# 设置MuJoCo渲染后端为EGL（用于图形渲染）
os.environ["MUJOCO_GL"] = "egl"

# 定义机器人关节名称列表 - 针对x1机器人的腿部关节
JOINT_NAMES = [
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee_pitch", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee_pitch", "right_ankle_pitch", "right_ankle_roll"
]

# 加载XML模型文件的绝对路径
xml_path = "./x1/mjcf/xyber_x1_flat.xml"
# 从XML文件创建MuJoCo模型
mjmodel = mujoco.MjModel.from_xml_path(xml_path)
# 获取每个关节在qpos数组中的索引位置
qpos_indices = np.array([mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in JOINT_NAMES])
# 创建MuJoCo数据实例，用于存储模拟状态
mjdata = mujoco.MjData(mjmodel)

# 定义关节控制增量（弧度制），用于键盘控制时的步进值
JOINT_INCREMENT = 0.05  # 可根据需要调整

# 定义关节限制范围（最小值和最大值，单位：弧度）
# 根据x1机器人的关节限制设置
control_qlimit = [
    [-3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14],  # 最小值
    [3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14, 3.14]  # 最大值
]

# 初始化目标关节位置（全部为0）
init_qpos = np.zeros(len(JOINT_NAMES))
target_qpos = init_qpos.copy()  # 复制初始关节位置作为目标位置

# 创建线程安全锁，用于多线程间的数据同步
lock = threading.Lock()

# 定义键盘映射：增加关节角度的按键（左腿）
key_to_joint_increase_left = {
    '1': 0,  # left_hip_pitch
    '2': 1,  # left_hip_roll
    '3': 2,  # left_hip_yaw
    '4': 3,  # left_knee_pitch
    '5': 4,  # left_ankle_pitch
    '6': 5,  # left_ankle_roll
}

# 定义键盘映射：减少关节角度的按键（左腿）
key_to_joint_decrease_left = {
    'q': 0,  # left_hip_pitch
    'w': 1,  # left_hip_roll
    'e': 2,  # left_hip_yaw
    'r': 3,  # left_knee_pitch
    't': 4,  # left_ankle_pitch
    'y': 5,  # left_ankle_roll
}

# 定义键盘映射：增加关节角度的按键（右腿）
key_to_joint_increase_right = {
    '7': 6,   # right_hip_pitch
    '8': 7,   # right_hip_roll
    '9': 8,   # right_hip_yaw
    '0': 9,   # right_knee_pitch
    '-': 10,  # right_ankle_pitch
    '=': 11,  # right_ankle_roll
}

# 定义键盘映射：减少关节角度的按键（右腿）
key_to_joint_decrease_right = {
    'u': 6,   # right_hip_pitch
    'i': 7,   # right_hip_roll
    'o': 8,   # right_hip_yaw
    'p': 9,   # right_knee_pitch
    '[': 10,  # right_ankle_pitch
    ']': 11,  # right_ankle_roll
}

# 字典用于跟踪当前按下的键及其方向（增加或减少）
keys_pressed = {}

# 处理按键按下事件的回调函数
def on_press(key):
    try:
        k = key.char.lower()  # 转换为小写以处理大小写输入
        if k in key_to_joint_increase_left:
            with lock:  # 使用锁确保线程安全
                keys_pressed[k] = 1  # 设置为增加方向
        elif k in key_to_joint_decrease_left:
            with lock:  # 使用锁确保线程安全
                keys_pressed[k] = -1  # 设置为减少方向
        elif k in key_to_joint_increase_right:
            with lock:  # 使用锁确保线程安全
                keys_pressed[k] = 1  # 设置为增加方向
        elif k in key_to_joint_decrease_right:
            with lock:  # 使用锁确保线程安全
                keys_pressed[k] = -1  # 设置为减少方向
        elif k == "z":
            # 重置为初始关节位置
            print(f'重置所有关节位置')
            with lock:  # 使用锁确保线程安全
                global target_qpos
                target_qpos = init_qpos.copy()  # 复制初始位置
        
    except AttributeError:
        pass  # 处理特殊键（如方向键等）

# 处理按键释放事件的回调函数
def on_release(key):
    try:
        k = key.char.lower()
        if k in keys_pressed:
            with lock:  # 使用锁确保线程安全
                del keys_pressed[k]  # 从按下键字典中移除该键
    except AttributeError:
        pass  # 处理特殊键

# 在单独的线程中启动键盘监听器
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

try:
    # 启动MuJoCo被动查看器（不控制模拟步进）
    with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
        
        start = time.time()  # 记录开始时间
        # 主循环：运行模拟直到查看器关闭或达到1000秒
        while viewer.is_running() and time.time() - start < 1000:
            step_start = time.time()  # 记录每一步的开始时间

            # 根据当前按下的键更新目标关节位置
            with lock:  # 使用锁确保线程安全
                for k, direction in keys_pressed.items():
                    if k in key_to_joint_increase_left:
                        joint_idx = key_to_joint_increase_left[k]
                        # 检查是否超过关节上限
                        if target_qpos[joint_idx] < control_qlimit[1][joint_idx]:
                            target_qpos[joint_idx] += JOINT_INCREMENT * direction
                    elif k in key_to_joint_decrease_left:
                        joint_idx = key_to_joint_decrease_left[k]
                        # 检查是否超过关节下限
                        if target_qpos[joint_idx] > control_qlimit[0][joint_idx]:
                            target_qpos[joint_idx] += JOINT_INCREMENT * direction  # direction是-1
                    elif k in key_to_joint_increase_right:
                        joint_idx = key_to_joint_increase_right[k]
                        # 检查是否超过关节上限
                        if target_qpos[joint_idx] < control_qlimit[1][joint_idx]:
                            target_qpos[joint_idx] += JOINT_INCREMENT * direction
                    elif k in key_to_joint_decrease_right:
                        joint_idx = key_to_joint_decrease_right[k]
                        # 检查是否超过关节下限
                        if target_qpos[joint_idx] > control_qlimit[0][joint_idx]:
                            target_qpos[joint_idx] += JOINT_INCREMENT * direction  # direction是-1

            # 打印当前目标关节角度
            print("Target qpos:", [f"{x:.3f}" for x in target_qpos])
            
            # 更新模拟中的关节位置
            mjdata.qpos[qpos_indices] = target_qpos  # 直接设置关节位置
            mujoco.mj_step(mjmodel, mjdata)  # 执行模拟步进

            # 更新查看器选项（例如，每秒切换接触点显示）
            with viewer.lock():  # 锁定查看器以进行更新
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(mjdata.time % 2)
            viewer.sync()  # 同步查看器状态
            
            # 时间管理，维持模拟时间步
            time_until_next_step = mjmodel.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)  # 等待直到下一个时间步

except KeyboardInterrupt:
    print("用户中断了模拟。")  # 用户中断模拟
finally:
    listener.stop()  # 停止键盘监听器
    viewer.close()  # 关闭查看器