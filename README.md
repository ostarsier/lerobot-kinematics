
# LeRobot-Kinematics: Simple and Accurate Forward and Inverse Kinematics Examples for the Lerobot SO100 ARM

## Declaration

This repository is a fork of the following projects:
- [Robotics Toolbox for Python](https://github.com/petercorke/robotics-toolbox-python)


It is part of the work on [Constrained Behavior Cloning for Robotic Learning](https://arxiv.org/abs/2408.10568?context=cs.RO).


## A. Installation (Ubuntu 20.04)

```bash
  git clone https://github.com/box2ai-robotics/lerobot-kinematics.git
  cd lerobot_kinematics
  pip install -e .
```

## B. Examples in simulation

We recommended to click on the terminal window with the mouse after startup and then enter the keys, to avoid that the keys in the departure mujoco change the configuration of the scene.

### (1) qpos control

Example of joint angle control, when opened the Mucojo visualization will appear and you can use the keyboard to control the corresponding angle change of the robot arm.

```shell
python examples/lerobot_keycon_qpos.py
```

- ``1, 2, 3, 4, 5, 6`` Increase the angle.

- ``q, w, e, r, t, y`` Decrease Angle.

Press and hold '0' to return to position


### (2) gpos Control

Example of Gripper Posture (gpos) control, where you can use the keyboard to control the end-posture changes of Lerobot in mucojo.

```shell
python examples/lerobot_keycon_gpos.py
```

| Key | Action +            | Key | Action -            |
|-----|---------------------|-----|---------------------|
| `w` | Move Forward        | `s` | Move Backward       |
| `a` | Move Right          | `d` | Move Left           |
| `r` | Move Up             | `f` | Move Down           |
| `e` | Roll +              | `q` | Roll -              |
| `t` | Pitch +             | `g` | Pitch -             |
| `z` | Gripper Open        | `c` | Gripper Close       |

Press and hold '0' to return to position


### (3) Genesis IK Control

Example of end-posture control, where you can use the keyboard to control the end-posture changes of Lerobot in mucojo.


```shell
python examples/lerobot_genesis.py
```


If this repository was helpful to you, please give us a little star and have a great time! ⭐ ⭐ ⭐ ⭐ ⭐

## 3. Examples in Real

Example of gripper posture (gpos) control, where you can use the keyboard to control the Lerobot's end posture changes in mucojo while going from simulation to physical control of a real Lerobot arm.

```shell
python examples/lerobot_keycon_gpos_real.py
```

If you're interested in this, you can try using the keyboard to collect data.
