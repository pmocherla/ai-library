#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 22:22:42 2022

@author: priyankamocherla
"""

from setuptools import setup

setup(
    name='classifiertrainer',
    version='0.1.0',    
    description='A simple KNN Training package for the Iris dataset',
    url='https://github.com/pmocherla/ai-library',
    author='Priyanka Mocherla',
    author_email='pmocherla@ntlworld.com',
    license='BSD 2-clause',
    packages=['classifiertrainer'],
    install_requires=['pandas>=1.3.4',
                      'scikit-learn>=0.24.2', 
                      'setuptools>=58.0.4',
                      'matplotlib>=3.4.3'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: MacOS',        
        'Programming Language :: Python :: 3.9',
    ],
)