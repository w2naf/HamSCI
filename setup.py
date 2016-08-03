#!/usr/bin/env python

from distutils.core import setup

setup(name='hamsci',
        version='0.0.1',
        description='Scientific Analysis Code for the Ham Radio Science Citizen Investigation',
        author='Nathaniel A. Frissell',
        author_email='nafrissell@vt.edu',
        url='http://www.hamsci.org',
        packages=['hamsci'],
        )
#Will also need to clone python-hamtools into home directory
#in home directory type the following: 
#	git clone https://github.com/n1ywb/python-hamtools.git
#Then run setup.py
