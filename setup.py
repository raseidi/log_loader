import os
from setuptools import setup

setup(
    name='log-loader',
    version='0.1',
    description='A tool for loading log data for ML tasks',
    long_description=open('README.md'),
    long_description_content_type='text/markdown',
    author='Rafael Seidi Oyamada',
    packages=['log_loader'],
    install_requirements=['setuptools', 'docopt'],
)