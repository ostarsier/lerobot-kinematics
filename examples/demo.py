from lerobot_kinematics.lerobot.lerobot_Kinematics import lerobot_FK
import numpy as np
from lerobot_kinematics.lerobot.lerobot_Kinematics import get_robot

# Get the SO100 robot model
robot = get_robot("so100")

# Define joint angles (in radians)
qpos = np.array([0.1, 0.0, 0.0, 0.0])  # All joints at zero position

# Compute forward kinematics
end_effector_pose = lerobot_FK(qpos, robot)

# Extract position and orientation
position = end_effector_pose[:3]  # [X, Y, Z]
orientation = end_effector_pose[3:]  # [gamma, beta, alpha]

print(f"Position: {position}")
print(f"Orientation: {orientation}")