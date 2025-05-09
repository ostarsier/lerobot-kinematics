# code by Boxjod LinCC111 2025.1.13 Box2AI-Robotics copyright 盒桥智能 版权所有

# 导入系统相关库
import os
import mujoco  # MuJoCo物理引擎库，用于机器人仿真
import mujoco.viewer  # MuJoCo可视化界面
import numpy as np  # 数值计算库
import time  # 时间控制
import threading  # 线程管理

# 导入机器人控制相关库
from lerobot_kinematics import lerobot_IK, lerobot_FK, get_robot  # 导入乐博特机器人运动学库，包含正逆运动学和机器人模型获取函数
from joyconrobotics import JoyconRobotics  # 导入Joy-Con控制器接口
import math  # 数学函数

# 设置numpy打印选项，使数组输出更易读
np.set_printoptions(linewidth=200)
# 设置MuJoCo使用EGL渲染后端
os.environ["MUJOCO_GL"] = "egl"

# 定义机器人关节名称列表
JOINT_NAMES = [
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow_pitch",
    "right_elbow_yaw",
    "right_wrist_pitch",
    "right_wrist_roll"
]

# 加载MuJoCo模型
xml_path = "./x1/mjcf/xyber_x1_right_arm.xml"  # X1右臂XML文件路径
mjmodel = mujoco.MjModel.from_xml_path(xml_path)  # 从XML文件创建MuJoCo模型
# 获取关节在qpos数组中的索引位置
qpos_indices = np.array([mjmodel.joint(name).id for name in JOINT_NAMES])
mjdata = mujoco.MjData(mjmodel)  # 创建MuJoCo数据实例，用于存储仿真状态

# 定义关节增量和位置增量常量
JOINT_INCREMENT = 0.01  # 关节角度增量（弧度）
POSITION_INSERMENT = 0.002  # 位置增量（米）

# 获取X1型号的机器人模型
robot = get_robot('x1')

# 定义控制限制范围，包含位置和角度的最小值和最大值
# 格式：[[x_min, y_min, z_min, roll_min, pitch_min, yaw_min], 
#        [x_max, y_max, z_max, roll_max, pitch_max, yaw_max]]
control_glimit = [[0.125, -0.4,  0.046, -3.1, -1.5, -1.5], 
                  [0.380,  0.4,  0.23,  3.1,  1.5,  1.5]]

# 初始化关节角度配置
init_qpos = np.zeros(len(JOINT_NAMES))  # 初始化为零位
target_qpos = init_qpos.copy()  # 目标关节角度，初始化为初始角度

# 计算初始末端执行器位姿（正运动学）
init_gpos = lerobot_FK(init_qpos, robot=robot)
target_gpos = init_gpos.copy()  # 目标末端执行器位姿，初始化为初始位姿

# 创建线程锁，用于多线程同步
lock = threading.Lock()
target_gpos_last = init_gpos.copy()  # 保存上一次的目标位姿
direction_data_r = [[], [], []]  # 用于存储方向数据

# 设置位置偏移量，使Joy-Con控制相对于初始位置
offset_position_m = init_gpos[0:3]

# 初始化右手Joy-Con控制器
joyconrobotics_right = JoyconRobotics(
    device="right",  # 使用右手Joy-Con
    horizontal_stick_mode='yaw_diff',  # 水平摇杆控制偏航差异
    close_y=True,  # 关闭Y轴控制
    limit_dof=True,  # 限制自由度
    glimit=control_glimit,  # 设置控制限制
    offset_position_m=offset_position_m,  # 位置偏移量
    # offset_euler_rad = offset_euler_rad,  # 欧拉角偏移量（已注释）
    common_rad=False,  # 不使用通用弧度
    lerobot=True,  # 使用乐博特机器人
    pitch_down_double=True  # 俯仰向下时速度加倍
)

# 初始化计时器
t = 0
try:
    # 启动MuJoCo被动查看器
    with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
        start = time.time()  # 记录开始时间
        # 主循环，运行1000秒或直到查看器关闭
        while viewer.is_running() and time.time() - start < 1000:
            step_start = time.time()  # 记录步骤开始时间
            
            # 第一帧初始化
            if t == 0:
                # 设置初始关节角度
                for i, joint_name in enumerate(JOINT_NAMES):
                    joint_id = mjmodel.joint(joint_name).id
                    mjdata.qpos[joint_id] = init_qpos[i]
                mujoco.mj_forward(mjmodel, mjdata)  # 更新前向动力学
                with viewer.lock():  # 锁定查看器进行更新
                    # 每隔一秒切换接触点显示
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(mjdata.time % 2)
                viewer.sync()  # 同步查看器
            
            t = t + 1  # 增加计时器
            
            # 获取Joy-Con控制器的输入
            target_pose, gripper_state_r, _ = joyconrobotics_right.get_control()
            
            # 打印目标姿态，保留3位小数
            print("target_pose:", [f"{x:.3f}" for x in target_pose])
            
            # 限制目标姿态在控制范围内
            for i in range(6):
                target_pose[i] = control_glimit[0][i] if target_pose[i] < control_glimit[0][i] else (control_glimit[1][i] if target_pose[i] > control_glimit[1][i] else target_pose[i])
    
            # 提取目标位置和姿态
            x_r = target_pose[0]  # X坐标
            z_r = target_pose[2]  # Z坐标
            _, _, _, roll_r, pitch_r, yaw_r = target_pose  # 提取欧拉角
            y_r = 0.01  # Y坐标固定为0.01
            
            # 组合右手目标位姿
            right_target_gpos = np.array([x_r, y_r, z_r, roll_r, pitch_r, yaw_r])
            
            # 计算逆运动学，获取关节角度
            qpos_inv, IK_success = lerobot_IK(mjdata.qpos[qpos_indices], right_target_gpos, robot=robot)
            
            # 如果逆运动学计算成功
            if IK_success:
                # 设置目标关节角度
                target_qpos = qpos_inv
                # 更新关节位置
                for i, joint_name in enumerate(JOINT_NAMES):
                    joint_id = mjmodel.joint(joint_name).id
                    mjdata.qpos[joint_id] = target_qpos[i]
                
                # 更新前向动力学
                mujoco.mj_forward(mjmodel, mjdata)
                with viewer.lock():  # 锁定查看器进行更新
                    # 每隔一秒切换接触点显示
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(mjdata.time % 2)
                viewer.sync()  # 同步查看器
                
                # 保存当前目标位姿
                target_gpos_last = right_target_gpos.copy()
            else:
                # 如果逆运动学失败，使用上一次成功的目标位姿
                right_target_gpos = target_gpos_last.copy()
                # 更新Joy-Con控制器的位置
                joyconrobotics_right.set_position(right_target_gpos[0:3])
            
            # 计算下一步之前的等待时间，保持稳定的仿真速率
            time_until_next_step = mjmodel.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            time.sleep(0.001)  # 额外的小延迟，确保稳定性
except KeyboardInterrupt:
    # 捕获键盘中断（Ctrl+C），关闭查看器
    viewer.close()