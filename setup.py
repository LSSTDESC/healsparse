from setuptools import setup, find_packages
from os import path

exec(open('healsparse/_version.py').read())

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'long_description.md'), encoding='utf-8') as f:
    long_description = f.read()

name = 'healsparse'

setup(
    name=name,
    packages=find_packages(exclude=('tests')),
    version=__version__, # noqa
    description='Sparse healpix maps and geometry library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Eli Rykoff, Javier Sanchez, and others',
    author_email='erykoff@stanford.edu',
    url='https://github.com/lsstdesc/healsparse',
    install_requires=['numpy', 'healpy', 'astropy'],
)
