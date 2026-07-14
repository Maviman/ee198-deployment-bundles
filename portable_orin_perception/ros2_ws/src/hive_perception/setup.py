from setuptools import find_packages, setup

package_name = "hive_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/perception.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Jordan Fronko",
    maintainer_email="themane213@gmail.com",
    description="Overhead ArUco perception: camera images -> arena-frame vehicle poses -> UDP pose frames for the pursuit controller.",
    license="Proprietary (EE198 course project)",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "aruco_detector = hive_perception.aruco_detector_node:main",
            "pose_bridge = hive_perception.pose_bridge_node:main",
            "calibrate_arena = hive_perception.calibrate_arena:main",
        ],
    },
)
