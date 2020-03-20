import sys
import struct
import time
import named_serial

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class dfbcardMM(object):

    def __init__(self, parent=None, addr=None, regIdx=None, chnIdx=None, stateIdx=None, bitIdx=None, width=None, value=0):
        
        super(dfbcardMM, self).__init__()
        
        self.COMMAND = '\033[95m'
        self.FCTCALL = '\033[94m'
        self.INIT = '\033[92m'
        self.WARNING = '\033[93m'
        self.FAIL = '\033[91m'
        self.ENDC = '\033[0m'
        self.BOLD = "\033[1m"

        self.serialport = named_serial.Serial(port='rack', shared = True)
        
        self.channels = 2
        self.states = 64
        self.card = "DFBx2"
        self.address = addr

        '''global defaults'''
                
        self.wreg6 = (6 << 25)
        self.wreg6_bitIdx = [0,8,12,16,20,21,24]
        self.wreg6_width = [8,4,4,4,1,3,1]
        self.wreg6_parameter = ["NSAMP","RLDneg","ARLsense","RLDpos","CLK","XPT","PS"]
        
        self.wreg7 = (7 << 25)
        self.wreg7_bitIdx = [0,8,14,18,22,23,24]
        self.wreg7_width = [8,6,4,4,1,1,1]
        self.wreg7_parameter = ["SETT","SEQLN","CARD DELAY","PROP DELAY","ST","LED","RST"]
        
        '''GPI defaults'''
                
        
        self.GPI1 = []
        self.GPI1 = (1 << 17)
        self.GPI1_bitIdx = [0,5]
        self.GPI1_width = [5,8]
        self.GPI1_parameter = ["CS","PHTR"]
                
        self.GPI2 = []
        self.GPI2 = (2 << 17)
        self.GPI2_bitIdx = [0,2]
        self.GPI2_width = [2,3]
        self.GPI2_parameter = ["DP","PLLC"]
                
        self.GPI3 = []
        self.GPI3 = (3 << 17)
        self.GPI3_bitIdx = [0,2,3]
        self.GPI3_width = [1,1,1]
        self.GPI3_parameter = ["SMenb","PR","PC"]
                
        self.GPI4 = (4 << 17)
        self.GPI4_bitIdx = 0
        self.GPI4_width = 1
        self.GPI4_parameter = "TM"
                
        self.GPI5 = (5 << 17)
        self.GPI5_bitIdx = 0
        self.GPI5_width = 16
        self.GPI5_parameter = "TPhi"
                
        self.GPI6 = (6 << 17)
        self.GPI6_bitIdx = 0
        self.GPI6_width = 16
        self.GPI6_parameter = "TPlo"
                
        self.GPI8 = (8 << 17)
        self.GPI8_bitIdx = [0]
        self.GPI8_width = [6]
        self.GPI8_parameter = "DSfr"
                
        self.GPI12 = (12 << 17)
        self.GPI12_bitIdx = 0
        self.GPI12_width = 2
        self.GPI12_parameter = "D1td"
                
        self.GPI13 = (13 << 17)
        self.GPI13_bitIdx = 0
        self.GPI13_width = 2
        self.GPI13_parameter = "D2td"
                
        self.GPI14 = (14 << 17)
        self.GPI14_bitIdx = 0
        self.GPI14_width = 2
        self.GPI14_parameter = "D3td"
                
        self.GPI15 = (15 << 17)
        self.GPI15_bitIdx = 0
        self.GPI15_width = 2
        self.GPI15_parameter = "D4td"
                
        self.GPI16 = (16 << 17)
        self.GPI16_bitIdx = 0
        self.GPI16_width = 14
        self.GPI16_parameter = "ARLsen"
                
        self.GPI17 = (17 << 17)
        self.GPI17_bitIdx = 0
        self.GPI17_width = 16
        self.GPI17_parameter = "RLDpos"
                
        self.GPI18 = (18 << 17)
        self.GPI18_bitIdx = 0
        self.GPI18_width = 16
        self.GPI18_parameter = "RLDneg"
        
        
        '''page index register default: CH 1, STATE 0'''
        
        self.wreg0 = (1 << 6)
                
        '''channel arrayed parameter defaults'''
        
        self.wreg4_default = (4 << 25)
        self.wreg4_bitIdx = [0,15,16,20,24]
        self.wreg4_width = [14,1,4,4,1]
        self.wreg4_parameter = ["TRIstep","GR","TRIstepsize","TRIdwell","TRInumsteps","TRIidx"]
        
        self.wreg4 = []
        
#         for i in range(self.channels):
#             self.wreg4.append(self.wreg4_default)
        
        '''state arrayed parameter defaults'''

        self.wreg1_default = (1 << 25)
        self.wreg1_bitIdx = [0]
        self.wreg1_width = [12]
        self.wreg1_parameter = ["ADClockpt"]
               
        self.wreg2_default = (2 << 25)
        self.wreg2_bitIdx = [0,16,17]
        self.wreg2_width = [14,1,1]
        self.wreg2_parameter = ["DAC A offset","TRIa","TRIb"]

        self.wreg3_default = (3 << 25)
        self.wreg3_bitIdx = [0,10,21,23,24]
        self.wreg3_width = [10,10,1,1,1]
        self.wreg3_parameter = ["I","P","ARL","FBb","FBa"]

        self.wreg5_default = (5 << 25)
        self.wreg5_bitIdx = [0,11]
        self.wreg5_width = [2,14]
        self.wreg5_parameter = ["Send Mode","DAC B offset"]

        self.wreg1_channel = []
        self.wreg2_channel = []
        self.wreg3_channel = []
        self.wreg5_channel = []
        
        self.wreg1 = []
        self.wreg2 = []
        self.wreg3 = []
        self.wreg5 = []
        
        for n in range(self.channels):
            for m in range(self.states):
                self.wreg1_channel.append(self.wreg1_default)
                self.wreg2_channel.append(self.wreg2_default)
                self.wreg3_channel.append(self.wreg3_default)
                self.wreg5_channel.append(self.wreg5_default)
            self.wreg1.append(self.wreg1_channel)
            self.wreg2.append(self.wreg2_channel)
            self.wreg3.append(self.wreg3_channel)
            self.wreg4.append(self.wreg4_default)
            self.wreg5.append(self.wreg5_channel)

#         for i in range(self.states):
#             self.wreg1.append(self.wreg1_default)    
#             self.wreg2.append(self.wreg2_default)    
#             self.wreg3.append(self.wreg3_default)    
#             self.wreg5.append(self.wreg5_default)    
        
        
    def updateGlbVal(self, regIdx=None, bitIdx=None, value=0):
        print("update DFB global")
        if regIdx == 6:
            for idx,val in enumerate(self.wreg6_bitIdx):
                if bitIdx == val:
                    print("WREG6: update",self.wreg6_parameter[idx])
                    if value > 2**self.wreg6_width[idx]-1:
                        print(self.FAIL + "parameter value overflows allotted register space", self.ENDC)
                        return
                    mask = 0xfffffff ^ ((2**self.wreg6_width[idx]-1) << bitIdx)
                    self.wreg6 = (self.wreg6 & mask) | (value << bitIdx)
                    print(self.wreg6, hex(self.wreg6))
                    return
            print(self.FAIL + "bit index for WREG6 invalid", self.ENDC)
        if regIdx == 7:
            for idx,val in enumerate(self.wreg6_bitIdx):
                if bitIdx == val:
                    print("WREG7: update",self.wreg7_parameter[idx])
                    if value > 2**self.wreg7_width[idx]-1:
                        print(self.FAIL + "parameter value overflows allotted register space", self.ENDC)
                        return
                    mask = 0xfffffff ^ ((2**self.wreg7_width[idx]-1) << bitIdx)
                    self.wreg7 = (self.wreg7 & mask) | (value << bitIdx)
                    print(self.wreg7, hex(self.wreg7))
                    return
            print(self.FAIL + "bit index for WREG6 invalid", self.ENDC)
        print(self.FAIL + "register index for", self.card,"card function call invalid", self.ENDC)
            

#             if bitIdx == 0:
#                 print "update NSAMP:", value
#                 mask = 0xfffff00
#             if bitIdx == 8:
#                 print "update RLDneg:", value
#                 mask = 0xffff0ff
#             if bitIdx == 12:
#                 print "update ARLsense:", value
#                 mask = 0xfff0fff
#             if bitIdx == 16:
#                 print "update RLDpos:", value
#                 mask = 0xff0ffff
#             if bitIdx == 20:
#                 print "update CLK boolean:", value
#                 mask = 0xfefffff
#             if bitIdx == 21:
#                 print "update XPT mode:", value
#                 mask = 0xf1fffff
#             if bitIdx == 24:
#                 print "update PS boolean:", value
#                 mask = 0xeffffff
#             self.wreg6 = (self.wreg6 & mask) | (value << bitIdx)
#             print self.wreg6, hex(self.wreg6), bin(self.wreg6)
#         if regIdx == 7:
#             if bitIdx == 0:
#                 print "update SETT:", value
#                 mask = 0xfffff00
#             if bitIdx == 8:
#                 print "update SEQLN:", value
#                 mask = 0xfffc0ff
#             if bitIdx == 14:
#                 print "update card delay:", value
#                 mask = 0xffc3fff
#             if bitIdx == 18:
#                 print "update propagation delay:", value
#                 mask = 0xfc3ffff
#             if bitIdx == 22:
#                 print "update ST boolean:", value
#                 mask = 0xfbfffff
#             if bitIdx == 23:
#                 print "update LED mode:", value
#                 mask = 0xf7fffff
#             if bitIdx == 24:
#                 print "update RST boolean:", value
#                 mask = 0xeffffff 
                
    def updateGlbReg(self, regIdx=None, value=0):
        if regIdx == 6:
            self.wreg6 = (self.wreg6 & 0xe000000) | value
            print("update WREG6:", self.wreg6)
            return
        if regIdx == 7:
            self.wreg7 = (self.wreg7 & 0xe000000) | value
            print("update WREG7:", self.wreg7)
            return
        print(self.FAIL + "register index for", self.card,"card GLOBAL invalid", self.ENDC)
                
    def updateGPIReg(self, regIdx=None, value=0):
        if regIdx == 8:
            if (value < 0) or (value > 0x3f):
                print(self.FAIL + "parameter value out of range", self.ENDC)
            else:
                self.GPI8 = (self.GPI8 & 0xffe0000) | value
                print("update GPI8:", self.GPI8_parameter, ":",self.GPI8)
            return
        if regIdx == 12:
            if (value < 0) or (value > 0x3):
                print(self.FAIL + "parameter value out of range", self.ENDC)
            else:
                self.GPI12 = (self.GPI12 & 0xffe0000) | value
                print("update GPI12:", self.GPI12_parameter, ":", self.GPI12)
            return
        if regIdx == 13:
            if (value < 0) or (value > 0x3):
                print(self.FAIL + "parameter value out of range", self.ENDC)
            else:
                self.GPI13 = (self.GPI13 & 0xffe0000) | value
                print("update GPI13:", self.GPI13_parameter, ":", self.GPI13)
            return
        if regIdx == 14:
            if (value < 0) or (value > 0x3):
                print(self.FAIL + "parameter value out of range", self.ENDC)
            else:
                self.GPI14 = (self.GPI14 & 0xffe0000) | value
                print("update GPI14:", self.GPI14_parameter, ":", self.GPI14)
            return
        if regIdx == 15:
            if (value < 0) or (value > 0x3):
                print(self.FAIL + "parameter value out of range", self.ENDC)
            else:
                self.GPI15 = (self.GPI15 & 0xffe0000) | value
                print("update GPI15:", self.GPI15_parameter, ":", self.GPI15)
            return
        print(self.FAIL + "register index for", self.card,"card GPI invalid", self.ENDC)
        
            
    def updatePage(self, chnIdx, stIdx):
        print(self.FCTCALL + "update WREG0 page register [CH/ST]:",chnIdx,"/",stIdx, self.ENDC)
        mask = 0xfffff00
        self.wreg0 = (self.wreg0 & mask) | (chnIdx << 6) | stIdx
        
                           
    def updateArrVal(self, chnIdx= 1, stIdx=0, regIdx=None, bitIdx=0, value=0):
        print(self.FCTCALL + "update DFB arrayed state parameter", self.ENDC)
        self.updatePage(chnIdx, stIdx)
        if regIdx == 1:
            for idx,val in enumerate(self.wreg1_bitIdx):
                if bitIdx == val:
                    print("WREG1: update",self.wreg1_parameter[idx])
                    if value > 2**self.wreg1_width[idx]-1:
                        print(self.FAIL + "parameter value overflows allotted register space", self.ENDC)
                        return
                    mask = 0xfffffff ^ ((2**self.wreg1_width[idx]-1) << bitIdx)
                    self.wreg1[chnIdx][stIdx] = (self.wreg1[chnIdx][stIdx] & mask) | (value << bitIdx)
                    print(self.wreg1[chnIdx][stIdx], hex(self.wreg1[chnIdx][stIdx]))
                    return
            print(self.FAIL + "bit index for WREG1 invalid", self.ENDC)
        if regIdx == 2:
            for idx,val in enumerate(self.wreg2_bitIdx):
                if bitIdx == val:
                    print("WREG2: update",self.wreg2_parameter[idx])
                    if value > 2**self.wreg2_width[idx]-1:
                        print(self.FAIL + "parameter value overflows allotted register space", self.ENDC)
                        return
                    mask = 0xfffffff ^ ((2**self.wreg2_width[idx]-1) << bitIdx)
                    self.wreg2[chnIdx][stIdx] = (self.wreg2[chnIdx][stIdx] & mask) | (value << bitIdx)
                    print(self.wreg2[chnIdx][stIdx], hex(self.wreg2[chnIdx][stIdx]))
                    return
            print(self.FAIL + "bit index for WREG2 invalid", self.ENDC)
        if regIdx == 3:
            for idx,val in enumerate(self.wreg3_bitIdx):
                if bitIdx == val:
                    print("WREG3: update",self.wreg3_parameter[idx])
                    if value > 2**self.wreg3_width[idx]-1:
                        print(self.FAIL + "parameter value overflows allotted register space", self.ENDC)
                        return
                    mask = 0xfffffff ^ ((2**self.wreg3_width[idx]-1) << bitIdx)
                    self.wreg3[chnIdx][stIdx] = (self.wreg3[chnIdx][stIdx] & mask) | (value << bitIdx)
                    print(self.wreg3[chnIdx][stIdx], hex(self.wreg3[chnIdx][stIdx]))
                    return
            print(self.FAIL + "bit index for WREG3 invalid", self.ENDC)
        if regIdx == 4:
            for idx,val in enumerate(self.wreg4_bitIdx):
                if bitIdx == val:
                    print("WREG4: update",self.wreg4_parameter[idx])
                    if value > 2**self.wreg4_width[idx]-1:
                        print(self.FAIL + "parameter value overflows allotted register space", self.ENDC)
                        return
                    mask = 0xfffffff ^ ((2**self.wreg4_width[idx]-1) << bitIdx)
                    self.wreg4[chnIdx] = (self.wreg4[chnIdx] & mask) | (value << bitIdx)
                    print(self.wreg4[chnIdx], hex(self.wreg4[chnIdx]))
                    return
            print(self.FAIL + "bit index for WREG4 invalid", self.ENDC)
        if regIdx == 5:
            for idx,val in enumerate(self.wreg5_bitIdx):
                if bitIdx == val:
                    print("WREG5: update",self.wreg5_parameter[idx])
                    if value > 2**self.wreg5_width[idx]-1:
                        print(self.FAIL + "parameter value overflows allotted register space", self.ENDC)
                        return
                    mask = 0xfffffff ^ ((2**self.wreg5_width[idx]-1) << bitIdx)
                    self.wreg5[chnIdx][stIdx] = (self.wreg5[chnIdx][stIdx] & mask) | (value << bitIdx)
                    print(self.wreg5[chnIdx][stIdx], hex(self.wreg5[chnIdx][stIdx]))
                    return
            print(self.FAIL + "bit index for WREG5 invalid", self.ENDC)
                
    def updateArrReg(self, chnIdx= 1, stIdx=None, regIdx=None, value=0):
        print(self.FCTCALL + "update DFB arrayed state register", self.ENDC)
        print("update WREG0 page register [CH/ST]:",chnIdx,"/",stIdx)
        mask = 0xfffff00
        self.wreg0 = (self.wreg0 & mask) | (chnIdx << 6) | stIdx
        if regIdx == 1:
            self.wreg1[stIdx] = (self.wreg1[stIdx] & 0xe000000) | value
            print("update WREG1:", self.wreg1[stIdx])
        if regIdx == 2:
            self.wreg2[stIdx] = (self.wreg2[stIdx] & 0xe000000) | value
            print("update WREG2:", self.wreg2[stIdx])
        if regIdx == 3:
            self.wreg3[stIdx] = (self.wreg3[stIdx] & 0xe000000) | value
            print("update WREG3:", self.wreg3[stIdx])
        if regIdx == 5:
            self.wreg5[stIdx] = (self.wreg5[stIdx] & 0xe000000) | value
            print("update WREG5:", self.wreg5[stIdx])
                
    def sendReg(self, wregval):
        print(self.COMMAND + "send to address", self.address, ":", self.BOLD, wregval, self.ENDC)
        b0 = (wregval & 0x7f ) << 1             # 1st 7 bits shifted up 1
        b1 = ((wregval >> 7) & 0x7f) <<  1     # 2nd 7 bits shifted up 1
        b2 = ((wregval >> 14) & 0x7f) << 1     # 3rd 7 bits shifted up 1
        b3 = ((wregval >> 21) & 0x7f) << 1     # 4th 7 bits shifted up 1
        b4 = (self.address << 1) + 1         # Address shifted up 1 bit with address bit set

        msg = struct.pack('BBBBB', b0, b1, b2, b3, b4)
        self.serialport.write(msg)
        

