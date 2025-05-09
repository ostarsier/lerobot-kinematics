from lerobot_kinematics.lerobot.lerobot_Kinematics import lerobot_FK
import numpy as np
from lerobot_kinematics.lerobot.lerobot_Kinematics import get_robot
from lerobot_kinematics.lerobot.lerobot_Kinematics import lerobot_IK

# 获取SO100机器人模型
robot = get_robot("so100")

# 设置初始关节角度
initial_joint_angles = np.array([0.0, 0.0, 0.0, 0.0])
print("初始关节角度:", initial_joint_angles)

# 使用正向运动学计算末端执行器位姿
initial_pose = lerobot_FK(initial_joint_angles, robot)
print("初始末端执行器位姿 [X, Y, Z, roll, pitch, yaw]:\n", initial_pose)

# 设置目标位姿 - 稍微改变末端执行器的位置
target_pose = initial_pose.copy()
target_pose[2] += 0.05  # Z方向增加5厘米
print("目标末端执行器位姿:\n", target_pose)

# 使用逆运动学计算达到目标位姿所需的关节角度
new_joint_angles, success = lerobot_IK(initial_joint_angles, target_pose, robot)

if success:
    print("so100 逆运动学计算成功!")
    print("计算得到的关节角度:", new_joint_angles)
    
    # 验证计算结果 - 使用计算得到的关节角度进行正向运动学计算
    verification_pose = lerobot_FK(new_joint_angles, robot)
    print("验证末端执行器位姿:\n", verification_pose)
    
    # 计算误差
    error = np.linalg.norm(verification_pose[:3] - target_pose[:3])
    print(f"位置误差: {error:.6f} 米")
else:
    print("so100 逆运动学计算失败，目标位姿可能超出机器人工作空间。")


print("----------------------------------------")
# 尝试使用x1机器人模型（如果已实现）
try:
    x1_robot = get_robot("x1")
    print("\n成功加载x1机器人模型")
    
    # 设置x1机器人的初始关节角度（7个关节）
    x1_initial_angles = np.zeros(7)
    
    # 使用正向运动学计算末端执行器位姿
    x1_initial_pose = lerobot_FK(x1_initial_angles, x1_robot)
    print("x1初始末端执行器位姿:\n", x1_initial_pose)
    
    # 设置目标位姿
    x1_target_pose = x1_initial_pose.copy()
    x1_target_pose[0] += 0.1  # X方向增加5厘米
    
    # 使用逆运动学计算关节角度
    x1_new_angles, x1_success = lerobot_IK(x1_initial_angles, x1_target_pose, x1_robot)
    
    if x1_success:
        print("x1逆运动学计算成功!")
        print("计算得到的关节角度:", x1_new_angles)
        
        # 验证计算结果 - 使用计算得到的关节角度进行正向运动学计算
        x1_verification_pose = lerobot_FK(x1_new_angles, x1_robot)
        print("验证末端执行器位姿:\n", x1_verification_pose)
        
        # 计算误差
        x1_error = np.linalg.norm(x1_verification_pose[:3] - x1_target_pose[:3])
        print(f"位置误差: {x1_error:.6f} 米")
    else:
        print("x1逆运动学计算失败")
        
except Exception as e:
    print(f"\n加载x1机器人模型时出错: {e}")
