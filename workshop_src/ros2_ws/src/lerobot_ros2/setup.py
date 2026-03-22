from setuptools import find_packages, setup

package_name = 'lerobot_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anton',
    maintainer_email='MrAnton07@mail.ru',
    description="ROS2 Action wrapper for LeRobot ACT policy",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "policy_action_server = lerobot_ros2.policy_action_server:main",
            "policy_action_client = lerobot_ros2.policy_action_client:main",
        ],
    },
)
