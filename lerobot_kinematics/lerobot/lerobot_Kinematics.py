import numpy as np
import math
from math import sqrt as sqrt
from spatialmath import SE3, SO3
from lerobot_kinematics.ET import ET
# 保留小数点后15位，并且在第15位后进行四舍五入
def atan2(first, second):
    return round(math.atan2(first, second), 15)

def sin(radians_angle):
    return round(math.sin(radians_angle), 15)

def cos(radians_angle):
    return round(math.cos(radians_angle), 15)

def acos(value):
    return round(math.acos(value), 15)

def round_value(value):
    return round(value, 15)

def get_robot():
    # to joint 1
    E1 = ET.tx(0.0612)
    E2 = ET.tz(0.0598)
    E3 = ET.Rz()
    
    # to joint 2
    E4 = ET.tx(0.02943)
    E5 = ET.tz(0.05504)
    E6 = ET.Ry()
    
    # to joint 3
    E7 = ET.tx(0.1127)
    E8 = ET.tz(-0.02798)
    E9 = ET.Ry()

    # to joint 4
    E10 = ET.tx(0.13504)
    E11 = ET.tz(0.00519)
    E12 = ET.Ry()
    
    # to joint 5
    E13 = ET.tx(0.0593)
    E14 = ET.tz(0.00996)
    E15 = ET.Rx()  
    # E16 = ET.Ry(np.pi / 2)
    
    E17 = ET.tx(0.09538)
    # to gripper
    # E16 = ET.Ry(np.pi / 2)
    
    so100 = E1*E2*E3*E4*E5*E6*E7*E8*E9*E10*E11*E12*E13*E14*E15*E17 # 

    return so100

PI = math.pi
so100 = get_robot()
#设置关节限位
so100.qlim = [[-2.2, -3.14158, -0.2,    -2.0,  -3.14158], 
              [2.2,   0.2,      3.14158, 1.8,  3.14158]]
def lerobot_FK(qpos_data):
    # 获取末端执行器的齐次变换矩阵（T为SE3对象）
    T = so100.fkine(qpos_data)
    
    # 提取位置 (X, Y, Z) — 使用SE3对象的属性
    X, Y, Z = T.t  # 直接使用t属性获取位置（X, Y, Z）
    
    # 提取旋转矩阵 (T.A) 和计算欧拉角 (alpha, beta, gamma)
    R = T.R  # 获取旋转部分（3x3矩阵）
    
    # 计算欧拉角
    beta = atan2(-R[2, 0], sqrt(R[0, 0]**2 + R[1, 0]**2))
    
    if cos(beta) != 0:  # 确保不除以零
        alpha = atan2(R[1, 0] / cos(beta), R[0, 0] / cos(beta))
        gamma = atan2(R[2, 1] / cos(beta), R[2, 2] / cos(beta))
    else:  # 当cos(beta)为零时，发生奇异情况
        alpha = 0
        gamma = atan2(R[0, 1], R[1, 1])
    
    return np.array([X, Y, Z, gamma, beta, alpha])
    
    
def lerobot_IK(q_now, target_pose):
    # x, y, z, roll, pitch, yaw = target_pose
    # r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)  # 欧拉角的顺序是 XYZ
    # R_mat = r.as_matrix()  # 获取旋转矩阵
    # T = np.eye(4)
    # T[:3, :3] = R_mat
    # T[:3, 3] = [x, y, z]
    R = SE3.RPY(target_pose[3:])
    print(R)
    T = SE3(target_pose[:3]) * R
    
    # T = self.robot.fkine(q_now)

    sol = so100.ikine_LM(Tep = T, q0=q_now, method= "sugihara")
    
    print("success:", sol.success)
    return sol.q

# def rtb_inverse_kinematics(self, q_now, target_pose):
    
#     tep = self.robot.fkine(q_now)
#     # """
#     # 目标末端位姿求解逆运动学，并解决过零和关节抖动问题
#     # target_pose: [x, y, z, roll, pitch, yaw]
#     # """
#     # x, y, z, roll, pitch, yaw = target_pose
#     # r = R.from_euler('xyz', [roll, pitch, yaw], degrees=False)  # 欧拉角的顺序是 XYZ
#     # R_mat = r.as_matrix()  # 获取旋转矩阵

#     # # 创建一个 4x4 的单位矩阵
#     # T = np.eye(4)
    
#     # # 将旋转矩阵放入 T 的左上 3x3 部分
#     # T[:3, :3] = R_mat
    
#     # # 将位移向量放入 T 的最后一列
#     # T[:3, 3] = [x, y, z]
    
#     # mask = [2, 2, 2, 0, 1, ] 
#     # # 使用 ikine_LM 方法求解逆运动学
#     sol = self.robot.ikine_LM(Tep=tep) # , q0=q_now[:5], mask=mask
    
#     # return sol.q
#     if sol.success:
#         # 如果 IK 解成功，获取解并平滑关节角度的变化
#         q = sol.q
#         q = self.smooth_joint_motion(q_now, q)
#         print("IK求解成功！")
#         return q
#     else:
#         print("IK求解失败！")
#         return None

def smooth_joint_motion(q_now, q_new):
    """
    对计算出的关节角度进行平滑处理，避免抖动和过零
    """
    # 获取当前的关节角度，假设初始角度保存在 self.robot.q
    q_current = q_now

    # 为了平滑运动，限制关节角度变化的幅度（可以设置一个最大增量）
    max_joint_change = 0.1  # 设置关节最大变化值，单位为弧度
    
    # 对每个关节进行平滑处理
    for i in range(len(q_new)):
        delta = q_new[i] - q_current[i]
        if abs(delta) > max_joint_change:
            # 如果角度变化大于最大增量，限制增量
            delta = np.sign(delta) * max_joint_change
        q_new[i] = q_current[i] + delta
    
    # 更新当前的关节角度
    self.robot.q = q_new
    return q_new