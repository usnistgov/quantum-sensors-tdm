#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages
import glob
# py_modules argument installs individual python files as top level modules
# which allows for example:
# `import zaber` instead of from `instruments import zaber`
# to match nist_qsp_readout all instruments and many files in nasa_client are py_modules
filesnames = glob.glob("instruments/*.py")
py_modules = [os.path.splitext(f)[0] for f in filenames]
py_modules.extend(['nasa_client/easyClient','nasa_client/easyClientNDFB','nasa_client/easyClientDastard',
'nasa_client/rpc_client_for_easy_client'])

setup(
    author="GCO",
    author_email='galen.oneil@nist.gov',
    python_requires='>=3.5',
    description="Software to help run a NIST TDM system with Python 3.",
    install_requires=["matplotlib", "numpy", "PyQt5","pySerial","LabJackPython"],
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
    py_modules = py_modules
)
å