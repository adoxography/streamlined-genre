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
    license='MIT',
    keywords='nlp genre classification speech low-resource languages',
    url='',
    install_requires=[
        'liac-arff==2.4.0',
        'librosa==0.7.2',
        'matplotlib==3.2.1',
        'nlpaug==0.0.14',
        'numpy==1.18.4',
        'scipy==1.4.1',
        'sklearn==0.0'
    ],
    long_description=read_file('README.md'),
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ]
)
