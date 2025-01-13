import os
import mujoco
import mujoco.viewer
import numpy as np
import time
from lerobot_kinematics import lerobot_IK, lerobot_FK
from pynput import keyboard
import threading
from collections import deque

# for joycon
from pyjoycon import GyroTrackingJoyCon, get_R_id, get_L_id, ButtonEventJoyCon, JoyCon
from glm import vec2, vec3, quat, angleAxis
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

# XML 模型的绝对路径
xml_path = "/home/aloha/data/robot/lerobot-kinematics/examples/mujoco/scene.xml"
mjmodel = mujoco.MjModel.from_xml_path(xml_path)
qpos_indices = np.array([mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in JOINT_NAMES])
mjdata = mujoco.MjData(mjmodel)

# 定义关节控制增量（弧度）
JOINT_INCREMENT = 0.01  # 可以根据需要调整
POSITION_INSERMENT = 0.002

# 定义关节限幅：
qlimit = [[-2.1, -3.14, -0.1, -2.0, -3.1, -0.1], 
          [2.1,   0.2,   3.14, 1.8,  3.1, 1.0]]
glimit = [[0.000, -0.4,  0.046, -3.1, -1.5, -1.5], 
          [0.430,  0.4,  0.23,  3.1,  1.5,  1.5]]

# 初始化目标关节位置
init_qpos = np.array([0.0, -3.14, 3.14, 0.0, 1.57, -0.157])
# init_qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
target_qpos = init_qpos.copy() # mjdata.qpos[qpos_indices].copy()
# init_gpos = rtb_kinematics.rtb_forward_kinematics(init_qpos[1:5])
init_gpos = lerobot_FK(init_qpos[1:5])
target_gpos = init_gpos.copy() 

# 线程安全锁
lock = threading.Lock()

###################################################################################################
###################################################################################################
target_gpos_last = init_gpos.copy()

def calc_target_distance(_target_gpos):
    x, y, z, r, p, y = _target_gpos
    distance = np.sqrt(x*x + y*y + z*z)
    # print(f'{distance=}')
    return distance

# 连接实体机械臂
motors = {"shoulder_pan":(1,"sts3215"),
        "shoulder_lift":(2,"sts3215"),
        "elbow_flex":(3,"sts3215"),
        "wrist_flex":(4,"sts3215"),
        "wrist_roll":(5,"sts3215"),
        "gripper":(6,"sts3215"),}
follower_arm = FeetechMotorsBus(port="/dev/ttyACM0", motors=motors,)
follower_arm.connect()
print(f'机械臂连接成功')

# 机械臂校正参数
arm_calib_path = "/home/boxjod/Genesis/box_arm/main_leader.json"
with open(arm_calib_path) as f:
    calibration = json.load(f)
follower_arm.set_calibration(calibration)

# 初始化数据容器
direction_data_r = [[], [], []]  # 分为三个子数据（假设direction包含三个值）
joycon_gyro_r, joycon_button_r, joycon_r, attitude_estimator_r = ConnectJoycon("right")
right_zero_pos = init_gpos[:3]
right_zero_euler = init_gpos[3:6]
x_r, y_r, z_r = init_gpos[0], init_gpos[1], init_gpos[2]
x0_r, y0_r, z0_r = init_gpos[0], init_gpos[1], init_gpos[2]
roll0_r, pitch0_r, yaw0_r = init_gpos[3], init_gpos[4], init_gpos[5]
gripper_state_r = 1
yaw_diff = 0

t = 0
try:
    with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < 1000:
            step_start = time.time()
            if t ==0 :
                mjdata.qpos[qpos_indices] = init_qpos
                mujoco.mj_step(mjmodel, mjdata)
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(mjdata.time % 2)
                viewer.sync()
                
            t = t+1
            attitude_estimator_value_r = attitude_estimator_r.update(joycon_gyro_r.gyro_in_rad[0], joycon_gyro_r.accel_in_g[0])
            roll_r, pitch_r, yaw_r = attitude_estimator_value_r[0], attitude_estimator_value_r[1], attitude_estimator_value_r[2]
            # pitch_r = -pitch_r # 手柄
            # yaw_r = -yaw_r * math.pi/2 #* 10
            yaw_r -= yaw_diff
            roll_r = roll_r + math.pi/2 # lerobo末端旋转90度
            
            if pitch_r > 0:
                pitch_r = pitch_r * 3.0
            
            yaw_rad_T = math.pi/2
            pitch_rad_T = math.pi/2
            pitch_r = pitch_rad_T if pitch_r > pitch_rad_T else (-pitch_rad_T/2 if pitch_r < -pitch_rad_T/2 else pitch_r) 
            yaw_r = yaw_rad_T if yaw_r > yaw_rad_T else (-yaw_rad_T if yaw_r < -yaw_rad_T else yaw_r) 
            
            direction_vector_r = vec3(math.cos(pitch_r) * math.cos(yaw_r), math.cos(pitch_r) * math.sin(yaw_r), math.sin(pitch_r))

            ########### 复位/夹爪按键 #############
            if joycon_r.get_button_home() == 1:
                joycon_gyro_r.reset_orientation
                attitude_estimator_r.reset_yaw()
                yaw_reset = target_qpos[0]
                yaw_diff = 0
                while 1:
                    ################right
                    x_r = x_r - 0.001 if x_r > right_zero_pos[0]+0.001 else (x_r + 0.001 if x_r < right_zero_pos[0]-0.001 else x_r) 
                    y_r = y_r - 0.001 if y_r > right_zero_pos[1]+0.001 else (y_r + 0.001 if y_r < right_zero_pos[1]-0.001 else y_r)
                    z_r = z_r - 0.001 if z_r > right_zero_pos[2]+0.001 else (z_r + 0.001 if z_r < right_zero_pos[2]-0.001 else z_r)
                    yaw_reset = yaw_reset - 0.001 if yaw_reset > 0.001 else (yaw_reset + 0.001 if yaw_reset < 0-0.001 else yaw_reset)
                    
                    right_target_gpos = np.array([x_r, y_r, z_r, roll_r, pitch_r, 0.0])
                    # qpos_inv = rtb_kinematics.rtb_inverse_kinematics(mjdata.qpos[qpos_indices][1:5], right_target_gpos)
                    qpos_inv = lerobot_IK(mjdata.qpos[qpos_indices][1:5], right_target_gpos)
                    target_qpos = np.concatenate(([yaw_reset,], qpos_inv, [gripper_state_r,]))
                    
                    mjdata.qpos[qpos_indices] = init_qpos
                    mujoco.mj_step(mjmodel, mjdata)
                    with viewer.lock():
                        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(mjdata.time % 2)
                    viewer.sync()
                    
                    if abs(x_r-right_zero_pos[0]) < 0.05 and abs(y_r-right_zero_pos[1]) < 0.05 and abs(z_r-right_zero_pos[2]) <0.05:
                        break
            
            for event_type, status in joycon_button_r.events():
                if event_type == 'plus' and status == 1:
                    joycon_gyro_r.calibrate()
                    joycon_gyro_r.reset_orientation
                    attitude_estimator_r.reset_yaw()
                    time.sleep(2)
                    joycon_gyro_r.calibrate()
                    joycon_gyro_r.reset_orientation
                    attitude_estimator_r.reset_yaw()
                    time.sleep(2)
                elif event_type == 'r':
                    if status == 1:
                        if gripper_state_r == 1:
                            gripper_state_r = -0.1
                        else:
                            gripper_state_r = 1
                        
                print(f'{event_type}')
               
            
            ########### 位移 #############
            joycon_stick_v_r = joycon_r.get_stick_right_vertical()
            if joycon_stick_v_r > 4000: # 向前移动：朝着方向矢量的方向前进 0.1 的速度
                x_r += 0.001 * direction_vector_r[0]
                z_r -= 0.001 * direction_vector_r[2]
            elif joycon_stick_v_r < 1000: # 向后移动：朝着方向矢量的反方向移动 0.1 的速度
                x_r -= 0.001 * direction_vector_r[0]
                z_r += 0.001 * direction_vector_r[2]

            joycon_stick_h_r = joycon_r.get_stick_right_horizontal()
            if joycon_stick_h_r > 4000 and yaw_diff < math.pi/2: 
                yaw_diff +=0.002
            elif joycon_stick_h_r < 1000 and yaw_diff > -math.pi/2: 
                yaw_diff -=0.002
            
            # joycon_button_r_up = joycon_r.get_button_r()
            # if joycon_button_r_up == 1:
            #     z_r += 0.0005
            
            # joycon_button_r_down = joycon_r.get_button_r_stick()
            # if joycon_button_r_down == 1:
            #     z_r -= 0.0005
                
            # 自定义按键 
            
            ########### 输出Joycon位姿 #############
            right_target_gpos = np.array([x_r, y_r, z_r, roll_r, pitch_r, 0.0])
            # print(f'{right_target_gpos=}')
            
            # qpos_inv = rtb_kinematics.rtb_inverse_kinematics(mjdata.qpos[qpos_indices][1:5], right_target_gpos)
            qpos_inv = lerobot_IK(mjdata.qpos[qpos_indices][1:5], right_target_gpos)
            if qpos_inv[0] != -1.0 and qpos_inv[1] != -1.0 and qpos_inv[2] != -1.0 and qpos_inv[3] != -1.0:
                # target_qpos = np.concatenate((target_qpos[0:1], qpos_inv, [gripper_state_r,])) # 使用遥杆控制yaw
                target_qpos = np.concatenate(([yaw_r,], qpos_inv, [gripper_state_r,])) # 使用陀螺仪控制yaw
                mjdata.qpos[qpos_indices] = target_qpos
                
                # 步进模拟
                mujoco.mj_step(mjmodel, mjdata)
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(mjdata.time % 2)
                viewer.sync()
                
                joint_angles = np.rad2deg(target_qpos)
                joint_angles[1] = -joint_angles[1]
                joint_angles[0] = -joint_angles[0]
                # joint_angles[5] = joint_angles[5]
                follower_arm.write("Goal_Position", joint_angles)
                position = follower_arm.read("Present_Position")
                
                target_gpos_last = right_target_gpos.copy() # 保存备份
            else:
                right_target_gpos = target_gpos_last.copy()
            
            # 时间管理以维持模拟时间步
            time_until_next_step = mjmodel.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            time.sleep(0.001)
except KeyboardInterrupt:
    print("用户中断了模拟。")
finally:
    follower_arm.disconnect()
