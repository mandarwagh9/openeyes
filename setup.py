from setuptools import setup
import os
from glob import glob

package_name = 'openeyes'

setup(
    name=package_name,
    version='3.0.1',
    packages=['src'],
    package_dir={'src': 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mandar Wagh',
    maintainer_email='mandar@example.com',
    description='Hardware-agnostic edge robot vision framework with world models',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'openeyes = src.main:main',
        ],
    },
)