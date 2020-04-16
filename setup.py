from setuptools import setup, find_packages

exec(open('healsparse/_version.py').read())

name = 'healsparse'

setup(
    name=name,
    packages=find_packages(exclude=('tests')),
    version=__version__, # noqa
    description='Sparse healpix maps and geometry library',
    author='Eli Rykoff and others',
    author_email='erykoff@stanford.edu',
    url='https://github.com/lsstdesc/healsparse',
    install_requires=['numpy', 'healpy', 'astropy'],
)
