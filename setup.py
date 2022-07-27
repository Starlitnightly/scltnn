#!/usr/bin/env python

r"""
Shim setup.py
"""

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

if __name__ == "__main__":
    setuptools.setup(
        name="scltnn",
        version='0.0.1',
        packages=['scltnn'],
        author='starlitnightly',
        author_email='starlitnightly@163.com',
        description='A library to calculate the latent time of scRNA-seq',
        long_description=long_description,  
        long_description_content_type="text/markdown",  
        url="https://github.com/Starlitnightly/scltnn.git",  # 模块github地址
        classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",],
        install_requires=[
            'scanpy',
            'scvelo',
            'scglue',
        ],
        python_requires='>=3',
    )