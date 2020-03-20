#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages


setup(
    author="GCO",
    author_email='galen.oneil@nist.gov',
    python_requires='>=3.5',
    description="Running a NIST TDM system.",
    install_requires=["matplotlib", "numpy", "PyQt5","pySerial","LabJackPython"],
    license="MIT license",
    include_package_data=True,
    keywords='realtime_gui',
    name='realtime_gui',
    packages=["realtime_gui"],
    test_suite='tests',
    url='https://github.com/ggggggggg/realtime_gui',
    version='0.1.0',
    zip_safe=False,
    package_data={'realtime_gui': ['ui/*.ui']},
    entry_points = {
        'console_scripts': ['realtime_gui=realtime_gui.realtime_gui:main'],
    }
)
å