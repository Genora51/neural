# -*- coding: utf-8 -*-

from setuptools import setup


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='neural',
    version='0.0.1',
    description='A library for feed-forward Neural Networks.',
    long_description=readme,
    author='Geno Racklin Asher',
    license=license,
    packages=['neural']
)
