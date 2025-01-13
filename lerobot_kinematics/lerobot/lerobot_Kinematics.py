# code by LinCC111 2025.1.13 Box2AI-Robotics copyright 盒桥智能 版权所有

import numpy as np
import math
from math import sqrt as sqrt
from spatialmath import SE3, SO3
from lerobot_kinematics.ET import ET
from scipy.spatial.transform import Rotation as R

# Retain 15 decimal places and round off after the 15th place
def atan2(first, second):
    return round(math.atan2(first, second), 5)

def sin(radians_angle):
    return round(math.sin(radians_angle), 5)

def cos(radians_angle):
    return round(math.cos(radians_angle), 5)

def acos(value):
    return round(math.acos(value), 5)

def round_value(value):
    return round(value, 5)

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
    
    E17 = ET.tx(0.09538)
    # to gripper
    
    so100 = E1 * E2 * E3 * E4 * E5 * E6 * E7 * E8 * E9 * E10 * E11 * E12 * E13 * E14 * E15 * E17 

    return so100

PI = math.pi
so100 = get_robot()

# Set joint limits
so100.qlim = [[-2.2, -3.14158, -0.2, -2.0, -3.14158], 
              [2.2, 0.2, 3.14158, 1.8, 3.14158]]

def lerobot_FK(qpos_data):
    # Get the end effector's homogeneous transformation matrix (T is an SE3 object)
    if len(qpos_data) != 5:
        print(f'{len(qpos_data)=}, Incorrect number of joints')
    T = so100.fkine(qpos_data)
    
    # Extract position (X, Y, Z) — use SE3 object's attribute
    X, Y, Z = T.t  # Directly use t attribute to get position (X, Y, Z)
    
    # Extract rotation matrix (T.A) and calculate Euler angles (alpha, beta, gamma)
    R = T.R  # Get the rotation part (3x3 matrix)
    
    # Calculate Euler angles
    beta = atan2(-R[2, 0], sqrt(R[0, 0]**2 + R[1, 0]**2))
    
    if cos(beta) != 0:  # Ensure no division by zero
        alpha = atan2(R[1, 0] / cos(beta), R[0, 0] / cos(beta))
        gamma = atan2(R[2, 1] / cos(beta), R[2, 2] / cos(beta))
    else:  # When cos(beta) is zero, singularity occurs
        alpha = 0
        gamma = atan2(R[0, 1], R[1, 1])
    
    return np.array([X, Y, Z, gamma, beta, alpha])
    
def lerobot_IK(q_now, target_pose):
    R = SE3.RPY(target_pose[3:])
    T = SE3(target_pose[:3]) * R
    
    sol = so100.ikine_LM(
            Tep=T, 
            q0=q_now,
            ilimit=10,  # 10 iterations
            slimit=2,  # 1 is the limit
            tol=1e-3)  # tolerance for convergence
    
    if sol.success:
        # If IK solution is successful, 
        return sol.q
    else:
        # If the target position is unreachable, IK fails
        return -1 * np.ones(len(q_now))

