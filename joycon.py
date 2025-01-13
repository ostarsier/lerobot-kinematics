import logging
import math
import struct
import threading
import time

from glm import vec3
# import hid
from joycon_robot import ConnectJoycon
from joycon_lerobot import FK, IK
import numpy as np


class JoyConController:
    def __init__(
        self,
        names,
        initial_position=None,
        *args,
        **kwargs,
    ):
        self.initial_position = np.array(initial_position) if initial_position else np.array([0.0, -3.14, 3.14, 0.0, 1.57, -0.157])
        self.target_qpos = self.initial_position.copy()
        self.initial_gpos = FK(self.initial_position[1:5])
        self.zero_pos = self.initial_gpos[:3]
        self.zero_euler = self.initial_gpos[3:]
        self.joycon_gyro, self.joycon_button, self.joycon, self.attitude_estimator = ConnectJoycon(names)
        self.target_qpos_last = None
    
    def get_command(self):
        attitude_estimator_value = self.attitude_estimator.update(self.joycon_gyro.gyro_in_rad[0],  self.joycon_gyro.accel_in_g[0])
        roll_r, pitch_r, yaw_r = attitude_estimator_value[0], attitude_estimator_value[1], attitude_estimator_value[2]
        pitch_r = -pitch_r # 手柄
        yaw_r = -yaw_r * math.pi/2 #* 10
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
        if self.joycon.get_button_home() == 1:
            self.joycon_gyro.reset_orientation
            self.attitude_estimator.reset_yaw()
            yaw_reset = self.target_qpos[0]
            yaw_diff = 0
            while 1:
                ################right
                x = x - 0.001 if x > self.zero_pos[0]+0.001 else (self.target_qpos + 0.001 if x < self.zero_pos[0]-0.001 else x) 
                y = y - 0.001 if y > self.zero_pos[1]+0.001 else (y + 0.001 if y < self.zero_pos[1]-0.001 else y)
                z = z - 0.001 if z > self.zero_pos[2]+0.001 else (z + 0.001 if z < self.zero_pos[2]-0.001 else z)
                yaw_reset = yaw_reset - 0.001 if yaw_reset > 0.001 else (yaw_reset + 0.001 if yaw_reset < 0-0.001 else yaw_reset)
                
                target_gpos = np.array([x, y, z, roll_r, pitch_r, 0.0])
                # qpos_inv = rtb_kinematics.rtb_inverse_kinematics(mjdata.qpos[qpos_indices][1:5], self.target_gpos)
                # qpos_inv = rtb_kinematics.rtb_inverse_kinematics(self.target_qpos[1:5], target_gpos)
                qpos_inv = IK(self.target_qpos[1:5], target_gpos)
                self.target_qpos = np.concatenate(([yaw_reset,], qpos_inv, [gripper_state,]))
                
                if abs(x-self.zero_pos[0]) < 0.05 and abs(y-self.zero_pos[1]) < 0.05 and abs(z-self.zero_pos[2]) <0.05:
                    break
                
        # 这里需要添加左joycon的监听事件
        for event_type, status in self.joycon_button.events():
            if event_type == 'plus' and status == 1:
                self.joycon_gyro.calibrate()
                self.joycon_gyro.reset_orientation
                self.attitude_estimator.reset_yaw()
                time.sleep(2)
                self.joycon_gyro.calibrate()
                self.joycon_gyro.reset_orientation
                self.attitude_estimator.reset_yaw()
                time.sleep(2)
            elif event_type == 'r':
                if status == 1:
                    if gripper_state == 1:
                        gripper_state = -0.1
                    else:
                        gripper_state = 1
                    
            print(f'{event_type}')
            
        
        ########### 位移 #############
        joycon_stick_v = self.joycon.get_stick_right_vertical()
        if joycon_stick_v > 4000: # 向前移动：朝着方向矢量的方向前进 0.1 的速度
            x += 0.001 * direction_vector_r[0]
            z -= 0.001 * direction_vector_r[2]
        elif joycon_stick_v < 1000: # 向后移动：朝着方向矢量的反方向移动 0.1 的速度
            x -= 0.001 * direction_vector_r[0]
            z += 0.001 * direction_vector_r[2]

        joycon_stick_h = self.joycon.get_stick_right_horizontal()
        if joycon_stick_h > 4000 and yaw_diff < math.pi/2: 
            yaw_diff +=0.002
        elif joycon_stick_h < 1000 and yaw_diff > -math.pi/2: 
            yaw_diff -=0.002
        
        # 自定义按键 
        ########### 输出Joycon位姿 #############
        self.target_gpos = np.array([x, y, z, roll_r, pitch_r, 0.0])

        # qpos_inv = rtb_kinematics.rtb_inverse_kinematics(self.target_qpos[1:5], self.target_gpos)
        qpos_inv = IK(self.target_qpos[1:5], self.target_gpos)
        if qpos_inv[0] != -1.0 and qpos_inv[1] != -1.0 and qpos_inv[2] != -1.0 and qpos_inv[3] != -1.0:
            self.target_qpos = np.concatenate(([yaw_r,], qpos_inv, [gripper_state,])) # 使用陀螺仪控制yaw
            
            joint_angles = np.rad2deg(self.target_qpos)
            joint_angles[1] = -joint_angles[1]
            joint_angles[0] = -joint_angles[0]
            # joint_angles[5] = joint_angles[5]
            self.target_qpos = joint_angles
            #仅需返回关节角度
            # follower_arm.write("Goal_Position", joint_angles)
            
            # position = follower_arm.read("Present_Position")
            
            self.target_gpos_last = self.target_gpos.copy() # 保存备份
        else:
            self.target_gpos = self.target_gpos_last.copy()
        
        # time.sleep(0.001)
        return self.target_qpos