from setuptools import setup, Extension
import os
import numpy

extra_folders = [
    "joycon_robotics/Kinematics/core",
]

def package_files(directory):
    paths = []
    for (pathhere, _, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", pathhere, filename))
    return paths


extra_files = []
for extra_folder in extra_folders:
    extra_files += package_files(extra_folder)

frne = Extension(
    "joycon_robotics.Kinematics.frne",
    sources=[
        "./joycon_robotics/Kinematics/core/vmath.c",
        "./joycon_robotics/Kinematics/core/ne.c",
        "./joycon_robotics/Kinematics/core/frne.c",
    ],
    include_dirs=["./joycon_robotics/Kinematics/core/"],
)

fknm = Extension(
    "joycon_robotics.Kinematics.fknm",
    sources=[
        "./joycon_robotics/Kinematics/core/methods.cpp",
        "./joycon_robotics/Kinematics/core/ik.cpp",
        "./joycon_robotics/Kinematics/core/linalg.cpp",
        "./joycon_robotics/Kinematics/core/fknm.cpp",
    ],
    include_dirs=["./joycon_robotics/Kinematics/core/", numpy.get_include()],
)

setup(
    ext_modules=[frne, fknm],
    package_data={"joycon_robotics.Kinematics": extra_files},
)