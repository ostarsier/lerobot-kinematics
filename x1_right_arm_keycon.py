#!/usr/bin/env python3

import mujoco
import numpy as np
import time
import threading
from pynput import keyboard

# 模型文件路径
xml_path = '/Users/shelbin/code/github/lerobot-kinematics/x1/mjcf/xyber_x1_right_arm.xml'

# 关节名称列表
joints = [
    'right_shoulder_pitch',
    'right_shoulder_roll',
    'right_shoulder_yaw',
    'right_elbow_pitch',
    'right_elbow_yaw',
    'right_wrist_pitch',
    'right_wrist_roll'
]

# 关节角度增量
angle_increment = 0.05

# 关节角度限制
angle_limits = {
    'min': -3.0,
    'max': 3.0
}

# 初始化关节角度
qpos = np.zeros(len(joints))

# 加载模型
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# 更新关节位置
def update_joint_positions():
    for i, joint_name in enumerate(joints):
        joint_id = model.joint(joint_name).id
        data.qpos[joint_id] = qpos[i]
    mujoco.mj_forward(model, data)

# 键盘映射
key_mapping = {
    # 增加关节角度的按键
    '1': (0, angle_increment),  # right_shoulder_pitch +
    '2': (1, angle_increment),  # right_shoulder_roll +
    '3': (2, angle_increment),  # right_shoulder_yaw +
    '4': (3, angle_increment),  # right_elbow_pitch +
    '5': (4, angle_increment),  # right_elbow_yaw +
    '6': (5, angle_increment),  # right_wrist_pitch +
    '7': (6, angle_increment),  # right_wrist_roll +
    
    # 减少关节角度的按键
    'q': (0, -angle_increment),  # right_shoulder_pitch -
    'w': (1, -angle_increment),  # right_shoulder_roll -
    'e': (2, -angle_increment),  # right_shoulder_yaw -
    'r': (3, -angle_increment),  # right_elbow_pitch -
    't': (4, -angle_increment),  # right_elbow_yaw -
    'y': (5, -angle_increment),  # right_wrist_pitch -
    'u': (6, -angle_increment),  # right_wrist_roll -
}

# 键盘事件处理
def on_press(key):
    global qpos
    try:
        key_char = key.char
        if key_char in key_mapping:
            joint_idx, increment = key_mapping[key_char]
            qpos[joint_idx] += increment
            # 限制关节角度范围
            qpos[joint_idx] = np.clip(qpos[joint_idx], angle_limits['min'], angle_limits['max'])
            print(f"关节 {joints[joint_idx]}: {qpos[joint_idx]:.2f}")
        elif key_char == 'z':
            # 重置所有关节位置
            qpos = np.zeros(len(joints))
            print("已重置所有关节位置")
    except AttributeError:
        pass

# 启动键盘监听
keyboard_listener = keyboard.Listener(on_press=on_press)
keyboard_listener.start()

# 主循环
try:
    # 使用mujoco_viewer如果可用，否则使用其他方法
    try:
        import mujoco_viewer
        viewer = mujoco_viewer.MujocoViewer(model, data)
        print("使用mujoco_viewer显示模型")
        print("\n键盘控制说明:")
        print("1-7: 增加对应关节角度")
        print("q-u: 减少对应关节角度")
        print("z: 重置所有关节位置")
        print("Ctrl+C: 退出程序\n")
        
        while viewer.is_alive:
            update_joint_positions()
            viewer.render()
            time.sleep(0.01)
        viewer.close()
    except ImportError:
        # 如果mujoco_viewer不可用，尝试使用mujoco.viewer
        try:
            from mujoco import viewer
            print("使用mujoco.viewer显示模型")
            print("\n键盘控制说明:")
            print("1-7: 增加对应关节角度")
            print("q-u: 减少对应关节角度")
            print("z: 重置所有关节位置")
            print("Ctrl+C: 退出程序\n")
            
            with viewer.launch_passive(model, data) as viewer:
                while True:
                    update_joint_positions()
                    viewer.sync()
                    time.sleep(0.01)
        except ImportError:
            # 如果两种方法都不可用，只打印关节角度
            print("无法加载可视化模块，只显示关节角度")
            print("\n键盘控制说明:")
            print("1-7: 增加对应关节角度")
            print("q-u: 减少对应关节角度")
            print("z: 重置所有关节位置")
            print("Ctrl+C: 退出程序\n")
            
            while True:
                update_joint_positions()
                time.sleep(0.1)
                
except KeyboardInterrupt:
    print("\n程序已退出")
finally:
    # 停止键盘监听
    keyboard_listener.stop()