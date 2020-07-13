#!/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = ['numpy', 'torch']
setup(name='sparselinear',
        version='0.0.5',
        description='Pytorch extension library for creating sparse linear layers',
        long_description=long_description,
        long_description_content_type="text/markdown",
        author='Rain Neuromorphics',
        author_email='ross@rain-neuromorphics.com',
        url='https://github.com/rain-neuromorphics/SparseLinear',
        keywords=['pytorch', 'sparse', 'linear'],
        license='MIT',
        install_requires=install_requires,
        packages=find_packages(),
        python_requires='>=3.6',
        )
