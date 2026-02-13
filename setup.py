from setuptools import find_packages, setup

package_name = 'kuka_training_multiple_factors_trajectories_pkg'

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
    maintainer='george',
    maintainer_email='georgeyaccoup124@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_sim = kuka_training_multiple_factors_trajectories_pkg.robot:main',
            'energy_monitor = kuka_training_multiple_factors_trajectories_pkg.Energy_clc:main',
            'ik_solver = kuka_training_multiple_factors_trajectories_pkg.inverse_kinematics_node:main',
            'trajectory_planner = kuka_training_multiple_factors_trajectories_pkg.input_trajectory:main',           
            'train_rl = kuka_training_multiple_factors_trajectories_pkg.train_rl:main',            
            'benchmark_collector = kuka_training_multiple_factors_trajectories_pkg.benchmark_collector:main', 
            'testing_model = kuka_training_multiple_factors_trajectories_pkg.test_model:main', 
            'model_status = kuka_training_multiple_factors_trajectories_pkg.final_exam:main',
            'model_status_graph = kuka_training_multiple_factors_trajectories_pkg.training_graph_generator_node:main', 
 
        ],
    },
)
