import ez_setup
ez_setup.use_setuptools()

from setuptools import setup, find_packages

exec(open('curveball/version.py').read()) # loads __version__

setup(
	name='curveball',
    version=__version__,
    author='yoavram',
    author_email='yoavram@gmail.com',
    url='https://yoavram.github.io/curveball',
    description='',
    long_description=open('README.md').read(),
    license='see LICENSE.txt',
    keywords="",
    packages= find_packages(exclude='docs')
)
