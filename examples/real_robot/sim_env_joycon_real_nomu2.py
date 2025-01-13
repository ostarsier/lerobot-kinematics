import os
import mujoco
import mujoco.viewer
import numpy as np
import time
from IK_SO100_box_gpos import rtb_Kinematics
import threading

# for joycon
from glm import vec3
from scipy.spatial.transform import Rotation as R
import math
from joycon_robot import ConnectJoycon

# for feetch
from feetech import FeetechMotorsBus
import numpy as np
import json

np.set_printoptions(linewidth=200)
# 设置 MuJoCo 渲染后端
os.environ["MUJOCO_GL"] = "egl"

# 定义关节名称
JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

# 初始化运动学
rtb_kinematics = rtb_Kinematics()

# 定义关节控制增量（弧度）
JOINT_INCREMENT = 0.01  # 可以根据需要调整
POSITION_INSERMENT = 0.002

# 定义关节限幅：
qlimit = [[-2.1, -3.14, -0.1, -2.0, -3.1, 0.0], 
          [2.1,   0.2,   3.14, 1.8,  3.1, 1.0]]
glimit = [[0.000, -0.4,  0.046, -3.1, -1.5, -1.5], 
          [0.430,  0.4,  0.23,  3.1,  1.5,  1.5]]

# 初始化目标关节位置
init_qpos = np.array([0.0, -3.0, 3.0, 0.14, 1.57, -0.157])

target_qpos_r = init_qpos.copy() # mjdata.qpos[qpos_indices].copy()
target_qpos_l = init_qpos.copy() # mjdata.qpos[qpos_indices].copy()
init_gpos = rtb_kinematics.rtb_forward_kinematics(init_qpos[1:5])
zero_pos = init_gpos[:3].copy()
target_gpos = init_gpos.copy() 

# 线程安全锁
lock = threading.Lock()

###################################################################################################
###################################################################################################

# 连接实体机械臂
motors = {"shoulder_pan":(1,"sts3215"),
        "shoulder_lift":(2,"sts3215"),
        "elbow_flex":(3,"sts3215"),
        "wrist_flex":(4,"sts3215"),
        "wrist_roll":(5,"sts3215"),
        "gripper":(6,"sts3215"),}

follower_arm_r = FeetechMotorsBus(port="/dev/ttyACM0", motors=motors,)
follower_arm_r.connect()
print(f'1.1 机械臂（右边）：连接成功')

follower_arm_l = FeetechMotorsBus(port="/dev/ttyACM1", motors=motors,)
follower_arm_l.connect()
print(f'1.2 机械臂（左边）：连接成功')

# 机械臂校正参数
arm_calib_path = "/home/boxjod/Genesis/box_arm/main_follower.json"
with open(arm_calib_path) as f:
    calibration = json.load(f)
follower_arm_r.set_calibration(calibration)
print(f'2.1 机械臂（右边）：校准数据读取完成')

arm_calib_path_l = "/home/boxjod/Genesis/box_arm/main_leader.json"
with open(arm_calib_path_l) as f:
    calibration_l = json.load(f)
follower_arm_l.set_calibration(calibration_l)
print(f'2.2 机械臂（左边）：校准数据读取完成')

# 初始化数据容器
joycon_gyro_r, joycon_button_r, joycon_r, attitude_estimator_r = ConnectJoycon("right")
x_r, y_r, z_r = init_gpos[0], init_gpos[1], init_gpos[2]
gripper_state_r = 1
yaw_diff_r = 0

joycon_gyro_l, joycon_button_l, joycon_l, attitude_estimator_l = ConnectJoycon("left")
x_l, y_l, z_l = init_gpos[0], init_gpos[1], init_gpos[2]
gripper_state_l = 1
yaw_diff_l = 0

def get_euler(attitude_estimator, joycon_gyro, yaw_diff):
    attitude_estimator_value = attitude_estimator.update(joycon_gyro.gyro_in_rad[0], joycon_gyro.accel_in_g[0])
    roll, pitch, yaw = attitude_estimator_value[0], attitude_estimator_value[1], attitude_estimator_value[2]
    yaw -= yaw_diff
    roll = roll + math.pi/2 # lerobo末端旋转90度
    pitch = pitch * 3.0 if pitch > 0 else pitch
    
    yaw_rad_T = math.pi/2
    pitch_rad_T = math.pi/2
    pitch = pitch_rad_T if pitch > pitch_rad_T else (-pitch_rad_T/2 if pitch < -pitch_rad_T/2 else pitch) 
    yaw = yaw_rad_T if yaw > yaw_rad_T else (-yaw_rad_T if yaw < -yaw_rad_T else yaw) 
    
    direction_vector = vec3(math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch))
    
    return roll, pitch, yaw, direction_vector

def get_robot_qpos(follower_arm):
    qpos_back = follower_arm.read("Present_Position")
    qpos_back[1] = -qpos_back[1]
    qpos_back[0] = -qpos_back[0]
    qpos_back = np.deg2rad(qpos_back)
    return qpos_back

def get_position(follower_arm, joycon, joycon_gyro, joycon_button, attitude_estimator, target_qpos, x, y, z, roll, pitch, yaw_diff, gripper_state, direction_vector):
    ########### 复位/夹爪按键 #############
    if (joycon.is_right() and joycon.get_button_home() == 1) or (joycon.is_left() and joycon.get_button_capture() == 1):
        joycon_gyro.reset_orientation
        attitude_estimator.reset_yaw()
        yaw_reset = target_qpos[0]
        yaw_diff = 0
        while 1:
            ################right
            x = x - 0.001 if x > zero_pos[0] + 0.001 else (x + 0.001 if x < zero_pos[0] - 0.001 else x_r) 
            y = y - 0.001 if y > zero_pos[1] + 0.001 else (y + 0.001 if y < zero_pos[1] - 0.001 else y_r)
            z = z - 0.001 if z > zero_pos[2] + 0.001 else (z + 0.001 if z < zero_pos[2] - 0.001 else z_r)
            yaw_reset = yaw_reset - 0.001 if yaw_reset > 0.001 else (yaw_reset + 0.001 if yaw_reset < -0.001 else yaw_reset)
            
            target_gpos = np.array([x, y, z, roll, pitch, 0.0])
            
            now_qpos_back = get_robot_qpos(follower_arm)
            qpos_inv = rtb_kinematics.rtb_inverse_kinematics(now_qpos_back[1:5], target_gpos)
            target_qpos = np.concatenate(([yaw_reset,], qpos_inv, [gripper_state,]))
            
            if abs(x-zero_pos[0]) < 0.05 and abs(y-zero_pos[1]) < 0.05 and abs(z-zero_pos[2]) <0.05:
                break
    
    for event_type, status in joycon_button.events():
        if (joycon.is_right() and event_type == 'plus' and status == 1) or (joycon.is_left() and event_type == 'minus' and status == 1):
            joycon_gyro.calibrate()
            joycon_gyro.reset_orientation
            attitude_estimator.reset_yaw()
            time.sleep(2)
            joycon_gyro.calibrate()
            joycon_gyro.reset_orientation
            attitude_estimator.reset_yaw()
            time.sleep(2)
        elif (joycon.is_right() and event_type == 'zr') or (joycon.is_left() and event_type == 'zl'):
            if status == 1:
                if gripper_state == 1:
                    gripper_state = 0.1
                else:
                    gripper_state = 1
    
    ########### 位移 #############
    joycon_stick_v = joycon.get_stick_right_vertical() if joycon.is_right() else joycon.get_stick_left_vertical()
    if joycon_stick_v > 4000: # 向前移动：朝着方向矢量的方向前进 0.1 的速度
        x += 0.001 * direction_vector[0]
        z -= 0.001 * direction_vector[2]
    elif joycon_stick_v < 1000: # 向后移动：朝着方向矢量的反方向移动 0.1 的速度
        x -= 0.001 * direction_vector[0]
        z += 0.001 * direction_vector[2]
    
    joycon_stick_h = joycon.get_stick_right_horizontal() if joycon.is_right() else joycon.get_stick_left_horizontal()
    if joycon_stick_h > 4000 and yaw_diff < math.pi/2: 
        yaw_diff +=0.002
    elif joycon_stick_h < 1000 and yaw_diff > -math.pi/2: 
        yaw_diff -=0.002
    
    # 自定义按键 
    joycon_button_up = joycon.get_button_r() if joycon.is_right() else joycon.get_button_l()
    if joycon_button_up == 1:
        z += 0.0005
        
    joycon_button_down = joycon.get_button_r_stick() if joycon.is_right() else joycon.get_button_l_stick()
    if joycon_button_down == 1:
        z -= 0.0005
    
    joycon_button_xup = joycon.get_button_x() if joycon.is_right() else joycon.get_button_up()
    joycon_button_xback = joycon.get_button_b() if joycon.is_right() else joycon.get_button_down()
    if joycon_button_xup == 1:
        x += 0.0005
    elif joycon_button_xback == 1:
        x -= 0.0005 

        
    return target_qpos, x, y, z, roll, pitch, yaw_diff, gripper_state
        
def write_robot(follower_arm, target_qpos):
    joint_angles = np.rad2deg(target_qpos)
    joint_angles[1] = -joint_angles[1]
    joint_angles[0] = -joint_angles[0]
    follower_arm.write("Goal_Position", joint_angles)
    
    # position = follower_arm.read("Present_Position")
##################################################################################
############################### 控制主循环 #########################################
##################################################################################

print(f'即可进行遥操作')
target_gpos_last_r = init_gpos.copy()
target_gpos_last_l = init_gpos.copy()
try:
    while(1):     
        # 更新数据
        roll_r, pitch_r, yaw_r, direction_vector_r = get_euler(attitude_estimator_r, joycon_gyro_r, yaw_diff_r)
        target_gpos_r, x_r, y_r, z_r, roll_r, pitch_r, yaw_diff_r, gripper_state_r = get_position(follower_arm_r, joycon_r, joycon_gyro_r, joycon_button_r, attitude_estimator_r, target_qpos_r, x_r, y_r, z_r, roll_r, pitch_r, yaw_diff_r, gripper_state_r, direction_vector_r)
        
        target_gpos_r = np.array([x_r, y_r, z_r, roll_r, pitch_r, 0.0])
        now_qpos_back_r = get_robot_qpos(follower_arm_r)
        qpos_inv_r = rtb_kinematics.rtb_inverse_kinematics(now_qpos_back_r[1:5], target_gpos_r)
        
        if qpos_inv_r[0] != -1.0 and qpos_inv_r[1] != -1.0 and qpos_inv_r[2] != -1.0 and qpos_inv_r[3] != -1.0:
            target_qpos_r = np.concatenate(([yaw_r,], qpos_inv_r, [gripper_state_r,])) # 使用陀螺仪控制yaw
            write_robot(follower_arm_r, target_qpos_r)
            target_gpos_last_r = target_qpos_r.copy() # 保存备份
        else:
            target_qpos_r = target_gpos_last_r.copy()
        

        roll_l, pitch_l, yaw_l, direction_vector_l = get_euler(attitude_estimator_l, joycon_gyro_l, yaw_diff_l)
        target_gpos_l, x_l, y_l, z_l, roll_l, pitch_l, yaw_diff_l, gripper_state_l = get_position(follower_arm_l, joycon_l, joycon_gyro_l, joycon_button_l, attitude_estimator_l, target_qpos_l, x_l, y_l, z_l, roll_l, pitch_l, yaw_diff_l, gripper_state_l, direction_vector_l)
        
        target_gpos_l = np.array([x_l, y_l, z_l, roll_l, pitch_l, 0.0])
        now_qpos_back_l = get_robot_qpos(follower_arm_l)
        qpos_inv_l = rtb_kinematics.rtb_inverse_kinematics(now_qpos_back_l[1:5], target_gpos_l)
        
        if qpos_inv_l[0] != -1.0 and qpos_inv_l[1] != -1.0 and qpos_inv_l[2] != -1.0 and qpos_inv_l[3] != -1.0:
            target_qpos_l = np.concatenate(([yaw_l,], qpos_inv_l, [gripper_state_l,])) # 使用陀螺仪控制yaw
            write_robot(follower_arm_l, target_qpos_l)
            target_gpos_last_l = target_gpos_l.copy() # 保存备份
        else:
            target_gpos_l = target_gpos_last_l.copy()
        
        time.sleep(0.01)
except KeyboardInterrupt:
    print("用户中断了模拟。")
finally:
    follower_arm_r.disconnect()
    follower_arm_l.disconnect()
