# read in csv files that match row address to bolo type and another with bolotype info

import os
import sys
import numpy as np
import pylab as pl
import csv
import pickle
from IPython import embed
from copy import copy

def main():
    r2btf = 'row2bolotypes.csv' # match row address to bolo type
    btf = 'bolotypes.csv' # matches a bolo type to bolo properties (leg length)

    #Convert csv files to lists
    btl = []
    with open(btf,encoding='utf-8-sig') as obtf:
        btcsv = csv.reader(obtf)
        btl = [cbb for cbb in btcsv]
    r2bl = []
    with open(r2btf,encoding='utf-8-sig') as or2btf:
        r2btcsv = csv.reader(or2btf)
        r2bl = [rrbb for rrbb in r2btcsv]


    # Make bolotype dictionary, use column headers as keys
    # when a value is a number, make it a float
    bt = {}
    for bb in btl[2:]: # skip 2 lines of header
        dic = {}
        for hh,dd in zip(btl[0],bb):
            if '\xef\xbb\xbf' in hh: # ugly characters in first cell
                hh = 'Split'

            # convert to int, float, or keep as string appropriately
            if dd.isdigit():
                dic[hh]=int(dd)
            elif dd.replace('.','',1).isdigit():
                dic[hh]=float(dd)
            else:
                dic[hh]=dd
        bt[int(bb[0])] = dic

    # make hwmap dictionary, use nn for column proxy
    hwmap = {}
    for nn,col in enumerate(r2bl[0][1::2]): # [A,B,C]
        hwmap[col] = {}
        for mm,rbb in enumerate(r2bl[1:]): # skip header, loop over rows
            #print rbb,col,rbb[2*nn],'2n+1',2*nn+1,rbb[2*nn+1]
            #print rbb[2*nn+1]
            if not rbb[2*nn+1]: # skip empty entries
                continue
            row = int(rbb[0])
            hwmap[col][row] = copy(bt[int(rbb[2*nn+1])]) # connect col,row to bolo type
            #embed();sys.exit()
            try:
                # Chip ID is Row#.Column#.Wafer#
                # separated by periods in a single cell of the csv file
                # attempt to split the strings into a list of integers
                hwmap[col][row]['chip id'] = [int(cc) for cc in rbb[2*nn+2].split('.')]
                # compute positions of chip ids
                # (R,C) = (0,0) is at lower left, which makes the center chip (9,9).
                # from edge of silicon to the opposite side's deep etch is 6.150 mm
                # this is square layout
                pitch = 6.15
                y,x,z = hwmap[col][row]['chip id'] # row controls y, col controls x!
                hwmap[col][row]['chip distance'] = pitch*np.sqrt((x-9)**2+(y-9)**2)
                hwmap[col][row]['chip position'] = pitch*np.array([x-9,y-9])
                ss = hwmap[col][row]['Split']
                #if ss > 11: ss-=12
                ss = ss%6
                dy = 0 #1.36 if ss>5 else -1.36 # top or bottom of chip
                dx = ((ss%6)-2.)*0.95-0.295 # bolo pitch is 0.95 with 0.295 mm offset
                #if ss>5: dx*=-1 # reverse "top" bolometers to go from right to left
                hwmap[col][row]['bolo position'] = pitch*np.array([x-9,y-9]) + np.array([dx,dy])
            except:
                hwmap[col][row]['chip id'] = rbb[2*nn+2] # add in chip id that comes from row2bolo data sheet

    for col in hwmap:
        for row in hwmap[col]:
            xx,yy=hwmap[col][row]['bolo position']
            pl.plot(xx,yy,'o')
            #print col,row,'{:.2f},{:.2f}'.format(xx,yy)


    #embed();sys.exit()
    with open('hwmap.pkl','wb') as ohw:
        pickle.dump(hwmap,ohw)
    print('hwmap.py successfully wrote hwmap.pkl')
if __name__=='__main__':
    main()
