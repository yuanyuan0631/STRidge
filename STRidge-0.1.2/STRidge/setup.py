# setup.py
from setuptools import setup, find_packages

setup(
    name='STRidge',
    version='0.1.2',
    py_modules=['STRidge'],
    install_requires=[
        'numpy',
    ],
    description='A method to determine the unknown terms of partial differential equations',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yuanyuan0631/STRidge.git',  # 项目的URL
    author='Yuan yuan',
    author_email='yuanyuan0631@gmail.com',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
