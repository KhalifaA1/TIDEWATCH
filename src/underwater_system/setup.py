from glob import glob
from setuptools import setup

setup(
    name='underwater_system', version='0.1.0',
    packages=['underwater_system'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/underwater_system']),
        ('share/underwater_system', ['package.xml']),
        ('share/underwater_system/launch', glob('launch/*.py')),
        ('share/underwater_system/worlds', glob('worlds/*.sdf')),
        ('share/underwater_system/worlds/meshes', glob('worlds/meshes/*')),
        ('share/underwater_system/config', glob('../../config/*')),
        ('share/underwater_system/dashboard', glob('dashboard/*')),
        ('share/underwater_system/scripts', glob('../../scripts/*.py')),
    ],
    install_requires=['setuptools', 'numpy', 'PyYAML'],
    maintainer='TIDEWATCH Team', maintainer_email='tidewatch@example.com',
    description='Decentralized underwater swarm simulation', license='Apache-2.0',
    entry_points={'console_scripts': [
        'tidewatch_node = underwater_system.ros_nodes:main',
        'tidewatch_dashboard = underwater_system.dashboard:main',
        'tidewatch_evaluate = underwater_system.evaluate:main',
    ]},
)
