#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0', 'spyder==5.1.5', 'numpy==1.20.3', 'opencv-python==4.1.2.30', 'matplotlib==3.3.1', 'mat73==0.46', 'scipy==1.6.2', 'joblib==1.0.1', 'scikit-image==0.18.1']

test_requirements = ['pytest>=3', ]

setup(
    author="Matthijs Gawehn",
    author_email='Matthijs.Gawehn@deltares.nl',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="on-the-flt estimation of coastal paramters from videos of a wave field",
    entry_points={
        'console_scripts': [
            'cocos_map=cocos_map.cli:main',
        ],
    },
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='cocos_map',
    name='cocos_map',
    packages=find_packages(include=['cocos_map', 'cocos_map.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/MatthijsGawehn/cocos_map',
    version='0.1.0',
    zip_safe=False,
)
