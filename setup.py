from setuptools import setup, find_packages

exec(open('healsparse/_version.py').read())

name = 'healsparse'

setup(
    name=name,
    packages=find_packages(exclude=('tests')),
    version=__version__,
    description='A sparse healpix implementation',
    author='Eli Rykoff and others',
    author_email='erykoff@stanford.edu',
    url='https://github.com/lsstdesc/healsparse',
)
