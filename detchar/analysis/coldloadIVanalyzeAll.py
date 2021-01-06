#! /usr/bin/env python3
''' coldloadIVanalyzeAll.py

script to analyze all rows of a coldload sweep and plot/save results.

usage:
./coldloadIVanalyzeAll.py <T_bath> <datafile.pkl> <rowMap.pkl> <calnums.pkl>

<rowMap.pkl> and <calnums.pkl> are optional
'''

import sys
import ivAnalyzer

N=len(sys.argv)
T = float(sys.argv[1])

if N==3:
    iv = ivAnalyzer.ivAnalyzer(sys.argv[2])
    for ii in range(iv.nrow):
        iv.analyzeColdload(0,ii,T,'all',nu=None,savePlots=True)
elif N==4:
    iv = ivAnalyzer.ivAnalyzer(sys.argv[2],sys.argv[3])
    for ii in range(iv.nrow):
        iv.analyzeColdload(0,ii,T,'all',nu='useRowMap',savePlots=True)
elif N==5:
    iv = ivAnalyzer.ivAnalyzer(sys.argv[2],sys.argv[3],sys.argv[4])
    for ii in range(iv.nrow):
        iv.analyzeColdload(0,ii,T,'all',nu='useRowMap',savePlots=True)
