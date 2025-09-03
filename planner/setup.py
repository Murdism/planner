from setuptools import find_packages, setup

package_name = 'planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # add  nodes
        ('share/' + package_name + '/nodes', ['planner/nodes/global_planner_node.py',
                                               'planner/nodes/local_planner_node.py',
                                               'planner/nodes/map_publisher.py']   ),  
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='av_mz',
    maintainer_email='muradsmebrahtu@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'global_planner_node = planner.nodes.global_planner_node:main',
            'local_planner_node = planner.nodes.local_planner_node:main',
            'map_publisher = planner.nodes.map_publisher:main',
        ],
    },
)
