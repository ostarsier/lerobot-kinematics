# code by LinCC111 Boxjod 2025.1.13 Box2AI-Robotics copyright 盒桥智能 版权所有

import numpy as np  # 导入NumPy库，用于数值计算和数组操作
import math  # 导入数学库，提供基本数学函数
from math import sqrt as sqrt  # 从math库导入平方根函数
from spatialmath import SE3, SO3  # 导入空间数学库，用于处理3D空间中的变换
from lerobot_kinematics.ET import ET  # 导入ET模块，用于构建机器人运动学模型
from scipy.spatial.transform import Rotation as R  # 导入旋转变换库，用于处理旋转矩阵和欧拉角

# 以下函数用于保留计算精度，将结果四舍五入到3位小数
# 二元反正切函数，返回结果保留3位小数
def atan2(first, second):
    return round(math.atan2(first, second), 3)

# 正弦函数，返回结果保留3位小数
def sin(radians_angle):
    return round(math.sin(radians_angle), 3)

# 余弦函数，返回结果保留3位小数
def cos(radians_angle):
    return round(math.cos(radians_angle), 3)

# 反余弦函数，返回结果保留3位小数
def acos(value):
    return round(math.acos(value), 3)

# 通用四舍五入函数，保留3位小数
def round_value(value):
    return round(value, 3)

# 创建SO100型号机器人的运动学模型
# 定义了机器人各关节之间的几何关系和关节限制。
def create_so100():
    # 注释掉的部分是到关节1的变换，当前实现中未使用
    # E1 = ET.tx(0.0612)  # X轴平移0.0612米
    # E2 = ET.tz(0.0598)  # Z轴平移0.0598米
    # E3 = ET.Rz()        # 绕Z轴旋转（关节1）
    
    # 到关节2的变换
    E4 = ET.tx(0.02943)  # X轴平移0.02943米
    E5 = ET.tz(0.05504)  # Z轴平移0.05504米
    E6 = ET.Ry()         # 绕Y轴旋转（关节2）
    
    # 到关节3的变换
    E7 = ET.tx(0.1127)    # X轴平移0.1127米
    E8 = ET.tz(-0.02798)  # Z轴平移-0.02798米
    E9 = ET.Ry()          # 绕Y轴旋转（关节3）

    # 到关节4的变换
    E10 = ET.tx(0.13504)  # X轴平移0.13504米
    E11 = ET.tz(0.00519)  # Z轴平移0.00519米
    E12 = ET.Ry()         # 绕Y轴旋转（关节4）
    
    # 到关节5的变换
    E13 = ET.tx(0.0593)   # X轴平移0.0593米
    E14 = ET.tz(0.00996)  # Z轴平移0.00996米
    E15 = ET.Rx()         # 绕X轴旋转（关节5）
    
    # 注释掉的部分是到夹爪的变换，当前实现中未使用
    # E17 = ET.tx(0.09538)  # X轴平移0.09538米到夹爪
    
    # 组合所有变换，构建完整的机器人运动学链
    so100 = E4 * E5 * E6 * E7 * E8 * E9 * E10 * E11 * E12 * E13 * E14 * E15  # E1 * E2 * E3 * 
    
    # 设置关节限制范围（弧度）
    # 第一行是各关节的最小值，第二行是各关节的最大值
    so100.qlim = [[-3.14158, -0.2,     -1.5, -3.14158], 
                  [ 0.2,      3.14158,  1.5,  3.14158]]
    
    return so100

# 获取指定型号的机器人
def get_robot(robot="so100"):
    # 目前仅支持SO100型号
    if robot == "so100":
        return create_so100()
    else:
        print(f"Sorry, we don't support {robot} robot now")
        return None

# 正向运动学函数：根据关节角度计算末端执行器的位置和姿态
def lerobot_FK(qpos_data, robot):
    # 检查输入关节角度的维度是否与机器人关节数量一致
    if len(qpos_data) != len(robot.qlim[0]):
        raise Exception("The dimensions of qpose_data are not the same as the robot joint dimensions")
    
    # 获取末端执行器的齐次变换矩阵（T是一个SE3对象）
    T = robot.fkine(qpos_data)
    
    # 提取位置（X, Y, Z）— 使用SE3对象的t属性
    X, Y, Z = T.t  # 直接使用t属性获取位置（X, Y, Z）
    
    # 提取旋转矩阵（T.R）并计算欧拉角（alpha, beta, gamma）
    R = T.R  # 获取旋转部分（3x3矩阵）

    # 计算欧拉角（使用ZYX欧拉角顺序）
    # beta是绕Y轴的旋转角度
    beta = atan2(-R[2, 0], sqrt(R[0, 0]**2 + R[1, 0]**2))
    
    # 避免除零错误，当cos(beta)不为零时计算alpha和gamma
    if cos(beta) != 0:  # 确保不会除以零
        alpha = atan2(R[1, 0] / cos(beta), R[0, 0] / cos(beta))  # 绕Z轴的旋转角度
        gamma = atan2(R[2, 1] / cos(beta), R[2, 2] / cos(beta))  # 绕X轴的旋转角度
    else:  # 当cos(beta)为零时，出现奇异性
        alpha = 0  # 设置alpha为0
        gamma = atan2(R[0, 1], R[1, 1])  # 计算gamma
    
    # 返回位置和姿态的组合数组 [X, Y, Z, gamma, beta, alpha]
    return np.array([X, Y, Z, gamma, beta, alpha])
    
# 逆向运动学函数：根据末端执行器的位置和姿态计算关节角度
def lerobot_IK(q_now, target_pose, robot):
    # 检查当前关节角度的维度是否与机器人关节数量一致
    if len(q_now) != len(robot.qlim[0]):
        raise Exception("The dimensions of qpose_data are not the same as the robot joint dimensions")
    
    # 注释掉的代码是使用spatialmath库的SE3.RPY方法创建变换矩阵的另一种方式
    # R = SE3.RPY(target_pose[3:])
    # T = SE3(target_pose[:3]) * R
    
    # 从目标位姿中提取位置和姿态
    x, y, z, roll, pitch, yaw = target_pose
    
    # 使用scipy.spatial.transform库创建旋转矩阵
    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)  # 欧拉角的顺序是 XYZ
    R_mat = r.as_matrix()  # 获取旋转矩阵
    
    # 创建4x4齐次变换矩阵
    T = np.eye(4)  # 创建单位矩阵
    T[:3, :3] = R_mat  # 设置旋转部分
    T[:3, 3] = [x, y, z]  # 设置平移部分
    
    # 使用Levenberg-Marquardt算法求解逆运动学
    sol = robot.ikine_LM(
            Tep=T,  # 目标位姿的齐次变换矩阵
            q0=q_now,  # 初始关节角度猜测值
            ilimit=10,  # 最大迭代次数为10
            slimit=2,  # 步长限制为2
            tol=1e-3)  # 收敛容差为0.001
    
    # 检查逆运动学求解是否成功
    if sol.success:
        # 如果逆运动学求解成功
        q = sol.q  # 获取求解的关节角度
        q = smooth_joint_motion(q_now, q, robot)  # 平滑关节运动
        # print(f'{q=}')  # 打印关节角度（已注释）
        return q, True  # 返回关节角度和成功标志
    else:
        # 如果目标位置不可达，逆运动学求解失败
        print(f'IK fails')  # 打印失败信息
        return -1 * np.ones(len(q_now)), False  # 返回全-1数组和失败标志
    
# 平滑关节运动函数：限制关节角度变化幅度，避免突变
def smooth_joint_motion(q_now, q_new, robot):
    q_current = q_now  # 当前关节角度
    max_joint_change = 0.1  # 最大关节角度变化量（弧度）
    
    # 遍历每个关节，限制角度变化
    for i in range(len(q_new)):
        delta = q_new[i] - q_current[i]  # 计算角度变化量
        if abs(delta) > max_joint_change:  # 如果变化量超过最大限制
            delta = np.sign(delta) * max_joint_change  # 限制变化量，保持方向不变
        q_new[i] = q_current[i] + delta  # 更新关节角度
    
    robot.q = q_new  # 更新机器人模型的关节角度
    return q_new  # 返回平滑后的关节角度
