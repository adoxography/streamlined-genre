#!/usr/bin/env python3
import os
from setuptools import setup


def read_file(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='streamlined-genre',
    version='0.0.1',
    author='Graham Still',
    author_email='gstill@uw.edu',
    description='',
    license='GPL-3.0',
    keywords='nlp genre classification speech low-resource languages',
    url='https://github.com/adoxography/streamlined-genre',
    install_requires=[
        'liac-arff==2.4.0',
        'librosa==0.7.2',
        'nlpaug==0.0.14',
        'numpy==1.18.4',
        'scipy==1.4.1',
        'sklearn==0.0'
    ],
    extras_require={
        'dev': [
            'flake8',
            'ipython',
            'pylint'
        ]
    },
    long_description=read_file('README.md'),
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ]
)
