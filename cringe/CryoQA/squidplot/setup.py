#!/usr/bin/env python

from distutils.core import setup

setup(name='squidplot',
      version='1.0',
      description='Simple squid data file plot',
      author='Gene Hilton',
      author_email='genehilton@gmail.com',
      packages=['squidplot'],
      package_dir={'squidplot': 'squidplot'},
      scripts = ['bin/squidplot']
     )
