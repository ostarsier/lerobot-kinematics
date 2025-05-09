import mujoco
import mujoco.viewer
import os

# 获取当前脚本所在的目录
# 构建 XML 文件的绝对路径
# 假设 xyber_x1_serial.xml 在脚本所在目录的 ./x1/mjcf/robot/xyber_x1/ 路径下
# 根据您的实际文件结构调整此路径
xml_path = "./x1/mjcf/xyber_x1_flat.xml"
# 如果 xml 文件在 /Users/shelbin/code/github/lerobot-kinematics/x1/mjcf/robot/xyber_x1/xyber_x1_serial.xml
# xml_path = "/Users/shelbin/code/github/lerobot-kinematics/x1/mjcf/robot/xyber_x1/xyber_x1_serial.xml"


if not os.path.exists(xml_path):
    print(f"错误：找不到模型文件 {xml_path}")
    print("请确保 xyber_x1_serial.xml 文件路径正确。")
    exit()

try:
    # 加载模型
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
except Exception as e:
    print(f"加载模型时出错: {e}")
    exit()

# 创建并启动查看器
try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 可以添加一些初始的相机设置或者控制逻辑
        # viewer.cam.azimuth = 150
        # viewer.cam.elevation = -20
        # viewer.cam.distance = 3.0
        # viewer.cam.lookat[:] = [0.0, 0.0, 0.55]

        print("查看器已启动。按 ESC 关闭。")
        while viewer.is_running():
            step_start = data.time

            # 在这里可以添加控制逻辑，例如设置关节角度
            # data.qpos[joint_id] = new_angle
            # data.ctrl[actuator_id] = control_value

            mujoco.mj_step(model, data)
            viewer.sync()

            # 如果需要，可以控制仿真速度
            # time_until_next_step = model.opt.timestep - (data.time - step_start)
            # if time_until_next_step > 0:
            #   time.sleep(time_until_next_step)

except Exception as e:
    print(f"启动查看器时出错: {e}")

print("程序结束。")