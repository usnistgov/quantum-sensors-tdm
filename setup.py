#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

setup(
    license="MIT license",
    include_package_data=True,
    keywords='tdm, tes',
    packages=find_packages(),
    test_suite='tests',
    url='',
    zip_safe=False,
    package_data={'': ['*.png', '*.ui']},
    entry_points={
        'console_scripts': ['adr_gui=adr_gui.adr_gui:main',
                            "cringe=cringe.cringe:main",
                            "tower_power_gui=instruments.tower_power_supply_gui:main",
                            "cringe_control=cringe.cringe_control:cringe_control_commandline_main",
                            "ls218_logger=instruments:_ls218_logger_entry_point"],
    },
    scripts=["doc/tdm_term"],
)
