
from setuptools import setup, find_packages
import os

setup(
    name='alf',
    version='0.0.6',
    install_requires=[
        'atari_py == 0.1.7',
        'cpplint',
        'clang-format == 9.0',
        'fasteners',
        'gin-config==0.4.0',
        'gym == 0.15.4',
        'gym3 == 0.3.3',
        'procgen == 0.10.4',
        'pyglet == 1.3.2',  # higher version breaks classic control rendering
        'matplotlib',
        'numpy',
        'opencv-python',
        'pathos == 0.2.4',
        # with python3.7, the default version of pillow (PIL) is 8.2.0,
        # which breaks some pyglet based rendering in gym
        'pillow==7.2.0',
        'psutil',
        'pybullet == 2.5.0',
        'rectangle-packer==2.0.0',
        'sphinx==3.0',
        'sphinx-autobuild',
        'sphinx-autodoc-typehints',
        'sphinxcontrib-napoleon==0.7',
        'sphinx-rtd-theme==0.4.3',  # used to build html docs locally
        'cnest',
    ],  # And any other dependencies alf needs
    package_data={'': ['*.gin']},
    packages=find_packages(),
)
