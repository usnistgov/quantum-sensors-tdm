#!/usr/bin/env python

"""The setup script."""

from setuptools import setup

setup(
    author="GCO",
    author_email='galen.oneil@nist.gov',
    python_requires='>=3.5',
    description="Software to help run a NIST TDM system with Python 3.",
    install_requires=["matplotlib", "numpy", "PyQt5","pySerial","LabJackPython", "lxml", 
    "argparse", "zmq", "scipy", "optparse"],
    license="MIT license",
    include_package_data=True,
    keywords='tdm, tes',
    name='qsptdm',
    packages=["adr_gui", "adr_system", "cringe", "named_serial", "nasa_client"],
    test_suite='tests',
    url='',
    version='0.1.0',
    zip_safe=False,
    package_data={'': ['*.png', '*.ui']},
    entry_points = {
        'console_scripts': ['adr_gui=adr_gui.adr_gui:main',
        "cringe=cringe.cringe:main"],
    },
)