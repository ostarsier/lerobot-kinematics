# code by LinCC111 Boxjod 2025.1.13 Box2AI-Robotics copyright 盒桥智能 版权所有

import os
import mujoco
import mujoco.viewer
import numpy as np
import time

from lerobot_kinematics import lerobot_IK, lerobot_FK, get_robot
from pynput import keyboard
import threading

# 设置NumPy打印选项，使数组输出更宽，便于查看
np.set_printoptions(linewidth=200)

# 设置MuJoCo渲染后端为EGL（用于图形渲染）
os.environ["MUJOCO_GL"] = "egl"

# 定义机器人关节名称列表
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

# 加载XML模型文件的绝对路径
xml_path = "./examples/scene.xml"
# 从XML文件创建MuJoCo模型
mjmodel = mujoco.MjModel.from_xml_path(xml_path)
# 获取每个关节在qpos数组中的索引位置
qpos_indices = np.array([mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in JOINT_NAMES])
# 创建MuJoCo数据实例，用于存储模拟状态
mjdata = mujoco.MjData(mjmodel)

# 定义关节控制增量（弧度制），用于键盘控制时的步进值
JOINT_INCREMENT = 0.01  # 可根据需要调整

# 创建机器人模型，指定型号为'so100'
robot = get_robot('so100')

# 定义关节限制范围（最小值和最大值，单位：弧度）
control_qlimit = [[-2.1, -3.1, -0.0, -1.4,    -1.57, -0.15], 
                  [ 2.1,  0.0,  3.1,  1.475,   3.1,  1.5]]

# 初始化目标关节位置
init_qpos = np.array([0.0, -3.14, 3.14, 0.0, -1.57, -0.157])
target_qpos = init_qpos.copy()  # 复制初始关节位置作为目标位置
# 计算初始关节位置对应的末端执行器位置（正向运动学）
init_gpos = lerobot_FK(init_qpos[1:5], robot=robot)

# 创建线程安全锁，用于多线程间的数据同步
lock = threading.Lock()

# 定义键盘映射：增加关节角度的按键
key_to_joint_increase = {
    '1': 0,  # Rotation（旋转关节）
    '2': 1,  # Pitch（俯仰关节）
    '3': 2,  # Elbow（肘部关节）
    '4': 3,  # Wrist_Pitch（腕部俯仰）
    '5': 4,  # Wrist_Roll（腕部旋转）
    '6': 5,  # Jaw（夹爪）
}

# 定义键盘映射：减少关节角度的按键
key_to_joint_decrease = {
    'q': 0,  # Rotation（旋转关节）
    'w': 1,  # Pitch（俯仰关节）
    'e': 2,  # Elbow（肘部关节）
    'r': 3,  # Wrist_Pitch（腕部俯仰）
    't': 4,  # Wrist_Roll（腕部旋转）
    'y': 5,  # Jaw（夹爪）
}

# 字典用于跟踪当前按下的键及其方向（增加或减少）
keys_pressed = {}

# 处理按键按下事件的回调函数
def on_press(key):
    try:
        k = key.char.lower()  # 转换为小写以处理大小写输入
        if k in key_to_joint_increase:
            with lock:  # 使用锁确保线程安全
                keys_pressed[k] = 1  # 设置为增加方向
        elif k in key_to_joint_decrease:
            with lock:  # 使用锁确保线程安全
                keys_pressed[k] = -1  # 设置为减少方向
        elif k == "0":
            # 重置为初始关节位置
            print(f'{k=}')
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
                    if k in key_to_joint_increase:
                        joint_idx = key_to_joint_increase[k]
                        # 检查是否超过关节上限
                        if target_qpos[joint_idx] < control_qlimit[1][joint_idx]:
                            target_qpos[joint_idx] += JOINT_INCREMENT * direction
                        
                    elif k in key_to_joint_decrease:
                        joint_idx = key_to_joint_decrease[k]
                        # 检查是否超过关节下限
                        if target_qpos[joint_idx] > control_qlimit[0][joint_idx]:
                            target_qpos[joint_idx] += JOINT_INCREMENT * direction  # direction是-1

            # 计算正向运动学，获取末端执行器位置
            position = lerobot_FK(target_qpos[1:5], robot=robot)
            print("Target qpos:", [f"{x:.3f}" for x in target_qpos])
            
            # 计算逆向运动学，根据末端执行器位置反推关节角度
            qpos_inv, ik_success = lerobot_IK(target_qpos[1:5], position, robot=robot)
            # 使用逆向运动学解决方案（如果有效）
            if ik_success:  # 检查逆向运动学解决方案是否有效
                # 更新目标关节位置，保留第一个关节和最后一个关节的值
                target_qpos = np.concatenate((target_qpos[0:1], qpos_inv[:4], target_qpos[5:]))

                # 更新模拟中的关节位置
                # mjdata.ctrl[qpos_indices] = target_qpos  # 使用控制器控制（已注释）
                mjdata.qpos[qpos_indices] = target_qpos  # 直接设置关节位置
                mujoco.mj_step(mjmodel, mjdata)  # 执行模拟步进

                # 更新查看器选项（例如，每秒切换接触点显示）
                with viewer.lock():  # 锁定查看器以进行更新
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(mjdata.time % 2)
                viewer.sync()  # 同步查看器状态
                
                target_gpos_last = target_qpos.copy()  # 备份有效的目标位置
            else:
                # 如果逆向运动学失败，恢复到上一个有效位置
                target_qpos = target_gpos_last.copy()
            
            # 时间管理，维持模拟时间步
            time_until_next_step = mjmodel.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)  # 等待直到下一个时间步

except KeyboardInterrupt:
    print("User interrupted the simulation.")  # 用户中断模拟
finally:
    listener.stop()  # 停止键盘监听器
    viewer.close()  # 关闭查看器
