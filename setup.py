#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

setup(
    author="GCO",
    author_email='galen.oneil@nist.gov',
    python_requires='>=3.5',
    description="Software to help run a NIST TDM system with Python 3.",
    install_requires=["matplotlib", "numpy", "PyQt5","pySerial","LabJackPython", "lxml", 
    "argparse", "zmq", "scipy","pytest", "pytest-qt", "ipython", "pyvisa", "progress",
    "dataclasses_json"],
    license="MIT license",
    include_package_data=True,
    keywords='tdm, tes',
    name='nistqsptdm',
    packages=find_packages(),
    test_suite='tests',
    url='',
    version='0.1.0',
    zip_safe=False,
    package_data={'': ['*.png', '*.ui']},
    entry_points = {
        'console_scripts': ['adr_gui=adr_gui.adr_gui:main',
        "cringe=cringe.cringe:main",
        "tower_power_gui=instruments.tower_power_supply_gui:main",
        "cringe_control=cringe.cringe_control:cringe_control_commandline_main"],
    },
    scripts = ["doc/tdm_term"],
)
