import numpy as np

import genesis as gs
gs.init(backend=gs.gpu)

######################################环境创建#######################################
dt = 0.02
scene = gs.Scene(
            sim_options= gs.options.SimOptions(dt = dt),
            viewer_options= gs.options.ViewerOptions(
                max_FPS= int(1 / dt),
                camera_pos= (0.0, 0.0, 2.5),
                camera_lookat= (0, 0, 0.3),
                camera_fov= 45,
            ),
            vis_options= gs.options.VisOptions(n_rendered_envs=1),
            rigid_options= gs.options.RigidOptions(
                dt= dt,
                constraint_solver= gs.constraint_solver.Newton,
                enable_collision= True,
                enable_joint_limit= True,
                enable_self_collision= True,
            ),
            show_viewer= True,
            # renderer= gs.renderers.RayTracer()
        )
#添加地面
plane = scene.add_entity(gs.morphs.Plane())

#添加灯光
light = scene.add_light(gs.morphs.Primitive()) 
#添加机械臂
so_100_right = scene.add_entity(gs.morphs.MJCF(file= 'sim_env/so_100.xml', pos=(-0.45, 0, 1.2), euler= (0, 0, 90)))
so_100_left = scene.add_entity(gs.morphs.MJCF(file= 'sim_env/so_100.xml', pos=(0.45, 0, 1.2), euler= (0, 0, -90)))
#添加桌子
# tablelegs = scene.add_entity(morph= gs.morphs.Mesh(file= 'sim_env/assets/table/tablelegs.obj', pos=(0, 0, 0), euler=(0, 0, 90), fixed= True),
#                              material= gs.materials.Rigid(friction= 0.9, rho=2000),
#                              visualize_contact= True,
#                               surface= gs.surfaces.Default(roughness= 0.7, diffuse_texture= gs.textures.ImageTexture(image_path= 'sim_env/assets/table/small_meta_table_diffuse.png')))

# tabletop = scene.add_entity( morph= gs.morphs.Mesh(file= 'sim_env/assets/table/tabletop.obj', pos=(0, 0, 0), euler=(0, 0, 90), fixed= True),
#                             material= gs.materials.Rigid(friction= 0.9, rho=2000, needs_coup= False),
#                             visualize_contact= True,
#                              surface= gs.surfaces.Default(roughness=0.7,  diffuse_texture= gs.textures.ImageTexture(image_path= 'sim_env/assets/table/small_meta_table_diffuse.png')))
table = scene.add_entity(morph= gs.morphs.Mesh(scale= (0.005, 0.005, 0.005),file= 'sim_env/assets/table/Desk.obj', pos=(0, 0, 0.99), euler=(90, 0, 0), fixed= True, collision= True), 
                         material= gs.materials.Rigid(),
                         visualize_contact= True,
                         surface= gs.surfaces.Metal(
                        diffuse_texture= gs.textures.ImageTexture(image_path= 'sim_env/assets/table/Top_Texture.jpg')))
#添加摄像头

#添加任务物体   #TODO 实现物体随机位置出现
#gs.materials.M
red_square = scene.add_entity(
                    material= gs.materials.Rigid(rho= 6250),
                    morph= gs.morphs.Box(pos=(0, 0, 1.2), size=(0.02, 0.02, 0.02), collision=True),
                    surface= gs.surfaces.Rough(
                        color= (1.0, 0.2, 0.2),
                        vis_mode= "visual",
                    ),
)
# cube = scene.add_entity(
#     morph=gs.morphs.Box(
#         size = (0.8, 0.8, 0.04),
#         pos  = (1.8, 0.0, 0.02),
#     ),
#     surface=gs.surfaces.Collision()
#     material=
#     # surface= gs.surfaces.Rough()
#             # color= (1.0, 0.2, 0.2),
#             # diffuse_texture= gs.textures.ImageTexture(image_path= 'sim_env/assets/table/small_meta_table_diffuse.png'))
# )

# cube2 = scene.add_entity(
#     gs.morphs.Box(
#         size = (0.04, 0.04, 0.04),
#         pos  = (1.8, 0.0, 0.4),
#     )
# )

# cube = scene.add_entity(
#     gs.morphs.Box(
#         size = (0.04, 0.04, 0.04),
#         pos  = (0.1, 0.2, 0.02),
#     )
# )

#环境创建
scene.build()
######################################################################################

#####################################机械臂控制#########################################
#机械臂关节名称，从机械臂xml文件获取
joint_names = [
    'Rotation',
    'Pitch',
    'Elbow',
    'Wrist_Pitch',
    'Wrist_Roll',
    'Jaw',
]
print(red_square.get_mass())
#设置机械臂初始位姿
left_joint_idx = [so_100_left.get_joint(name).dof_idx_local for name in joint_names]
right_joint_idx = [so_100_right.get_joint(name).dof_idx_local for name in joint_names]

#直接关节角控制
#初始位姿
init_pos = np.array([0, -3.14, 3.14, 0.817, 0, -0.157])
so_100_left.set_dofs_position(init_pos, left_joint_idx)
so_100_right.set_dofs_position(init_pos, right_joint_idx)

#PD控制
#机械臂PD控制参数和力、力矩范围限制
kp = np.array([2500, 2500, 1500, 1500, 800, 100])
kv = np.array([250, 250, 150, 150, 80, 10])
force_upper = np.array([50, 50, 50, 50, 12, 100])
force_lower = np.array([-50, -50, -50, -50, -12, -100])
#左臂
so_100_left.set_dofs_kp(kp= kp, dofs_idx_local= left_joint_idx)
so_100_left.set_dofs_kv(kv= kv, dofs_idx_local= left_joint_idx)
so_100_left.set_dofs_force_range(lower= force_lower, upper= force_upper, dofs_idx_local= left_joint_idx)
#右臂
so_100_right.set_dofs_kp(kp= kp, dofs_idx_local= right_joint_idx)
so_100_right.set_dofs_kv(kv =kv, dofs_idx_local= right_joint_idx)
so_100_right.set_dofs_force_range(lower= force_lower, upper= force_upper, dofs_idx_local= right_joint_idx)

# while(True):
#     for i in range(1250):
#         if i == 200:
#             so_100_left.control_dofs_position(np.array([0, -PI, PI, 0.817, 0, -0.157]), left_joint_idx)
#             so_100_right.control_dofs_position(np.array([0, -PI, PI, 0.817, 0, -0.157]), right_joint_idx)
#         elif i== 500:
#             so_100_left.control_dofs_position(np.array([0, -PI/2, PI/2, 0.66, 0, -0.157]), left_joint_idx)
#             so_100_right.control_dofs_position(np.array([0, -PI/2, PI/2, 0.66, 0, -0.157]), right_joint_idx)
#         elif i == 1000:
#             so_100_left.control_dofs_position(np.array([0, -PI, PI, 0.817, 0, -0.157]), left_joint_idx)
#             so_100_right.control_dofs_position(np.array([0, -PI, PI, 0.817, 0, -0.157]), right_joint_idx)
#         scene.step()    

#逆运动学控制
so_100_left.control_dofs_position(np.array([0, -3.14, 3.14, 0.817, 0, -0.157]), left_joint_idx)
scene.step()
left_end_effector = so_100_left.get_link('Fixed_Jaw')
left_trajectory = [

]


while(True):
    
    episode_len = 400
    for i in range(episode_len):
        print(red_square.get_dofs_force())
        if i < 200:
            left_target_pos = np.array([0.1, 0.1, 1.35])
            left_target_quat = np.array([0.707, 0.707, 0, 0])
            t_frac = i / 200
            cur_pos = np.array(left_end_effector.get_pos().cpu())
            cur_quat = np.array(left_end_effector.get_quat().cpu())
            print(cur_pos, cur_quat)
            next_pos = cur_pos + (left_target_pos - cur_pos) * t_frac
            next_quat = cur_quat + (left_target_quat - cur_quat) * t_frac
            next_qpos = so_100_left.inverse_kinematics(
                link= left_end_effector,
                pos = next_pos,
                quat = next_quat
            )
            next_qpos[-1] = 1.5
            so_100_left.control_dofs_position(next_qpos, left_joint_idx)
            scene.step()
        elif i < 250:
            left_target_pos = np.array([0.1, 0.1, 1.3])
            left_target_quat = np.array([0.707, 0.707, 0, 0])
            t_frac = (i - 200) / (400 - 200)
            cur_pos = np.array(left_end_effector.get_pos().cpu())
            cur_quat = np.array(left_end_effector.get_quat().cpu())
            print(cur_pos)
            next_pos = cur_pos + (left_target_pos - cur_pos) * t_frac
            next_quat = cur_quat + (left_target_quat - cur_quat) * t_frac
            next_qpos = so_100_left.inverse_kinematics(
                link= left_end_effector,
                pos = next_pos,
                quat = next_quat
            )
            next_qpos[-1] = -0.18
            so_100_left.control_dofs_position(next_qpos, left_joint_idx)
            scene.step()