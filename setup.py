#!/usr/bin/env python3
import io
import os
from setuptools import find_packages, setup


NAME = 'streamlined-genre'
DESCRIPTION = ''
URL = 'https://github.com/adoxography/streamlined-genre'
EMAIL = 'gstill@uw.edu'
AUTHOR = 'Graham Still'
REQUIRES_PYTHON = '>=3.6.0'

REQUIRED = [
    'liac-arff==2.4.0',
    'librosa==0.7.2',
    'nlpaug==0.0.14',
    'numpy==1.18.4',
    'scipy==1.4.1',
    'sklearn==0.0',
    'wget==3.2'
]

EXTRAS = {
    'dev': [
        'flake8',
        'ipython',
        'mypy',
        'pylint'
    ]
}

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as readme_file:
    long_description = '\n' + readme_file.read()

about = {}
with open(os.path.join(here, 'genre', '__version__.py')) as version_file:
    exec(version_file.read(), about)  # pylint: disable=exec-used
version = about['__version__']

setup(
    name=NAME,
    version=version,
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(),
    license='GPL-3.0',
    keywords='nlp genre classification speech low-resource languages',
    url=URL,
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    long_description=long_description,
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ]
)
