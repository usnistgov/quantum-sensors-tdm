#-*- coding: utf-8 -*-
import sys
import optparse
import struct
import time
import pickle
import os.path

from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import numpy as np
from .tuneclient import TuneClient
from .muxmaster import MuxMaster
from . import analysis
from . import vphistats
from cringe import log

def find(condition):
    res, = np.nonzero(np.ravel(condition))
    return res

class TuneTab(QWidget):
    def __init__(self, parent):
        super(type(self),self).__init__(parent)
        self.mm = MuxMaster(parent)
        self.layout = QVBoxLayout(self)

        self.c = TuneClient(self)
        self.layout.addWidget(self.c)

        hline = QFrame()
        hline.setFrameStyle(QFrame.HLine)
        self.layout.addWidget(hline)

        self.vphidemo = VPhiDemo(self,self.mm, self.c)
        self.layout.addWidget(self.vphidemo)

        hline = QFrame()
        hline.setFrameStyle(QFrame.HLine)
        self.layout.addWidget(hline)

        self.biterrordemo = BitErrorDemo(self,self.mm, self.c)
        self.layout.addWidget(self.biterrordemo)

        hline = QFrame()
        hline.setFrameStyle(QFrame.HLine)
        self.layout.addWidget(hline)

        self.settSweep = SettlingTimeSweeper(self, self.mm, self.c)
        self.layout.addWidget(self.settSweep)

        hline = QFrame()
        hline.setFrameStyle(QFrame.HLine)
        self.layout.addWidget(hline)

        self.biasSweeper = BiasSweeper(self, self.mm, self.c)
        self.layout.addWidget(self.biasSweeper)
        self._test_check_true = True

    def packState(self):
        stateVector = {}
        stateVector["ISlopeProduct"] = self.vphidemo.ISlopeProductSpin.value()
        stateVector["IMixProduct"] = self.vphidemo.MixSlopeProductSpin.value()
        stateVector["PercentFromBottomOfVphi"] = self.vphidemo.PercentFromBottomSpin.value()
        stateVector["minimumD2aAValue"] = self.vphidemo.minimumD2AValueSpin.value()
        stateVector["lockSlopeSignCheckBoxChecked"] = self.vphidemo.lockSlopeSignCheckBox.isChecked()
        stateVector["lockSlopeSignFBBCheckBoxChecked"] = self.vphidemo.lockSlopeSignFBBCheckBox.isChecked()
        return stateVector


    def unpackState(self, loadState):
        self.vphidemo.ISlopeProductSpin.setValue(loadState["ISlopeProduct"])
        self.vphidemo.MixSlopeProductSpin.setValue(loadState["IMixProduct"])
        self.vphidemo.PercentFromBottomSpin.setValue(loadState["PercentFromBottomOfVphi"])
        self.vphidemo.minimumD2AValueSpin.setValue(loadState.get("minimumD2aAValue",200))
        self.vphidemo.lockSlopeSignCheckBox.setChecked(loadState.get("lockSlopeSignCheckBoxChecked",False))
        self.vphidemo.lockSlopeSignFBBCheckBox.setChecked(loadState.get("lockSlopeSignFBBCheckBoxChecked",True))


class VPhiDemo(QWidget):
    def __init__(self, parent, mm, client):
        super(type(self), self).__init__(parent)
        self.layout = QVBoxLayout(self)
        self.mm=mm
        self.c=client

        self.layout.addWidget(QLabel("vphis and tuning"))

        layout = QHBoxLayout()
        self.button = QPushButton(self,text="vphis (changes mix to zero, may change send mode)")
        self.button.clicked.connect(self.oneOffVphi)
        layout.addWidget(self.button)

        self.vphi_type_combo = QComboBox()
        self.vphi_type_combo.addItem('unlocked FBA')
        self.vphi_type_combo.addItem('unlocked FBB')
        self.vphi_type_combo.addItem('locked FBA')
        layout.addWidget(self.vphi_type_combo)
        self.layout.addLayout(layout)
        self.vphi_functions = [self.FBAvphi, self.FBBvphi, self.lockedFBAvphi]


        layout = QHBoxLayout()


        self.fulltunebutton = QPushButton(self,text="full tune")
        self.fulltunebutton.clicked.connect(self.fullTune)
        layout.addWidget(self.fulltunebutton)



        layout2 = QHBoxLayout()
        self.sendmixcheckbox = QCheckBox(self)
        self.sendmixcheckbox_label = QLabel("check box to send mix after full tune")
        self.sendmixcheckbox.setChecked(True)
        layout2.addWidget(self.sendmixcheckbox)
        layout2.addWidget(self.sendmixcheckbox_label)
        layout.addLayout(layout2)

        self.learn_columns_button = QPushButton(self,text="learn columns")
        self.learn_columns_button.clicked.connect(self.learnColumns)
        layout.addWidget(self.learn_columns_button)

        self.zero_mix_button = QPushButton(self,text="set Mix=Zero")
        self.zero_mix_button.clicked.connect(self.c.client.setMixToZero)
        layout.addWidget(self.zero_mix_button)

        self.prune_bad_button = QPushButton(self,text="prune bad")
        self.prune_bad_button.clicked.connect(self.prune_bad_channels)
        layout.addWidget(self.prune_bad_button)

        self.layout.addLayout(layout)

        layout = QHBoxLayout()
        self.ISlopeProductSpin = QSpinBox()
        self.ISlopeProductSpin.setRange(-1000,1000)
        self.ISlopeProductSpin.setValue(-150)
        layout.addWidget(QLabel("I*Slope product"))
        layout.addWidget(self.ISlopeProductSpin)
        self.lockSlopeSignCheckBox = QCheckBox("Lock on + Slope", self)
        layout.addWidget(self.lockSlopeSignCheckBox)
        self.lockSlopeSignFBBCheckBox = QCheckBox("Lock on + Slope FBB",self)
        self.layout.addWidget(self.lockSlopeSignFBBCheckBox)
        self.layout.addLayout(layout)

        layout = QHBoxLayout()
        self.MixSlopeProductSpin = QSpinBox()
        self.MixSlopeProductSpin.setRange(-1000,1000)
        self.MixSlopeProductSpin.setValue(-400)
        layout.addWidget(QLabel("Mix*Slope product"))
        layout.addWidget(self.MixSlopeProductSpin)
        self.layout.addLayout(layout)

        layout = QHBoxLayout()
        self.PercentFromBottomSpin = QSpinBox()
        self.PercentFromBottomSpin.setRange(1,99)
        self.PercentFromBottomSpin.setValue(18)
        layout.addWidget(QLabel("Lockpoint % from bottom of vphi"))
        layout.addWidget(self.PercentFromBottomSpin)
        self.layout.addLayout(layout)

        layout = QHBoxLayout()
        self.minimumD2AValueSpin = QSpinBox()
        self.minimumD2AValueSpin.setRange(0,16383)
        self.minimumD2AValueSpin.setValue(200)
        layout.addWidget(QLabel("Minimum D2A Setpoint"))
        layout.addWidget(self.minimumD2AValueSpin)
        self.layout.addLayout(layout)

        layout = QHBoxLayout()
        grabd2abutton = QPushButton(self,text="grab d2aA values, assumes feedback is on")
        grabd2abutton.clicked.connect(self.grab_and_set_d2aA_values)
        plotnoisebutton = QPushButton(self,text="plot noise (changes nothing)")
        plotnoisebutton.clicked.connect(self.plotnoise)
        layout.addWidget(grabd2abutton)
        layout.addWidget(plotnoisebutton)
        self.layout.addLayout(layout)


    def learnColumns(self):
        """Set the mix off. For each DFBx2 card channel, set the feedback to a fixed known value.
        Then take data, and inspect the data to figure out which column maps to which DFBx2 card channel.
        Directly modify muxmaster to adopt this info.
        mm.gatherAllCards() will "repair" muxmaster to again use all cards in order from left to right
        """
        self.c.client.setMixToZero()
        self.mm.gatherAllCards()
        for i,dfbrap in enumerate(self.mm.dfbraps):
            self.mm.setdfballrow(col=i,d2aA=i)
        data = self.c.getNewData(0.1,minimumNumPoints=1)
        fba = data[0,0,:,1] #feedback
        err = data[0,0,:,0] #error
        np.save("last_learn_columns_data",data)
        new_dfbraps = []
        for col in range(data.shape[0]):
            vals = data[col,:,:,1]
            val=int(vals[0,0])
            assert (vals==val).all
            new_dfbrap = self.mm.dfbraps[val]
            new_dfbraps.append(new_dfbrap)
            log.info(("learned col %g reads %s"%(col,repr(new_dfbrap))))

        self.mm.dfbraps = new_dfbraps

    def oneOffVphi(self):

        vphi_type_index = self.vphi_type_combo.currentIndex()
        vphi_function = self.vphi_functions[vphi_type_index]
        self.c.client.setMixToZero()
        outtriangle, outsigsup, outsigsdown = vphi_function()



        plots = ColPlots(self,self.c.ncol,self.c.nrow)
        plots.plot(outtriangle,outsigsup)
        plots.title("one off vphi")
        plots.xlabel("triangle")
        plots.ylabel("signal")
        plots.show()


    def FBAvphi(self):
        tridwell,tristeps,tristepsize=2,9,10
        self.mm.settriangleparams(tridwell,tristeps,tristepsize)
        self.mm.setdfball(tria=1)
        data = self.c.getNewData(0.1,minimumNumPoints=4096*6)
        fba = data[0,0,:,1] #triangle
        err = data[0,0,:,0] #signal
        np.save("last_fba_vphi",data)

        outtriangle, outsigsup, outsigsdown = analysis.conditionvphis(data[:,:,:,1], data[:,:,:,0], tridwell, tristeps, tristepsize)
        return outtriangle, outsigsup, outsigsdown

    def FBBvphi(self):
        tridwell, tristeps, tristepsize = 2, 9, 20
        self.mm.settriangleparams(tridwell, tristeps, tristepsize)
        self.mm.setdfball(trib=1, data_packet=1) # Triangle feedback on FB[B], SendMode : FBB, ERR
        data = self.c.getNewData(0.1, minimumNumPoints=4096*6)
        fbb = data[0, 0, :, 1] # triangle
        err = data[0, 0, :, 0] # signal
        np.save("last_fbb_vphi", data)

        outtriangle, outsigsup, outsigsdown = analysis.conditionvphis(data[:,:,:,1], data[:,:,:,0], tridwell, tristeps, tristepsize)
        return outtriangle, outsigsup, outsigsdown

    def shouldSendMixAfterFullTune(self):
        return self.sendmixcheckbox.isChecked()

    def lockedFBAvphi(self, d2aB=8000, a2d = 1200, I=10):
        tridwell,tristeps,tristepsize=2,9,20
        self.mm.settriangleparams(tridwell,tristeps,tristepsize)
        self.mm.setdfball(tria=1,ARL=1,data_packet=2, FBB=1, I=I, d2aB=d2aB, a2d=a2d)

        data = self.c.getNewData(0.1,minimumNumPoints=4096*6,sendMode=2)
        fba = data[0,0,:,1] #signal
        fbb = data[0,0,:,0] #triangle
        np.save("last_locked_fba_vphi",data)

        outtriangle, outsigsup, outsigsdown = analysis.conditionvphis(data[:,:,:,0], data[:,:,:,1], tridwell, tristeps, tristepsize)
        return outtriangle, outsigsup, outsigsdown

    def lockedFBAvphi_colsettings(self, d2aB, a2d, I):
        tridwell,tristeps,tristepsize=2,9,15
        self.mm.settriangleparams(tridwell,tristeps,tristepsize)

        # for col in range(self.c.ncol):
        #     self.mm.setdfballrow(col,tria=1,ARL=1,data_packet=2, FBB=0, I=I[col], d2aB=d2aB[col], a2d=a2d[col])
        for col in range(self.c.ncol):
            self.mm.setdfballrow(col,tria=1,ARL=1,data_packet=2, FBB=1, I=I[col], d2aB=d2aB[col], a2d=a2d[col])


        data = self.c.getNewData(0.1,minimumNumPoints=4096*6,sendMode=2)
        fba = data[0,0,:,1] #signal
        fbb = data[0,0,:,0] #triangle
        np.save("last_locked_fba_vphi",data)

        outtriangle, outsigsup, outsigsdown = analysis.conditionvphis(data[:,:,:,0], data[:,:,:,1], tridwell, tristeps, tristepsize)
        return outtriangle, outsigsup, outsigsdown

    def validateServerSettings(self):
        """
        return True if nrow, nsamp, lsync all match between the server and cringe
        return False Otherwise, also pop-up a warning
        """
        returnVal = True
        if not self.c.nrow == self.mm.seqln:
            returnVal = False
        if not self.c.nsamp == self.mm.NSAMP:
            returnVal = False
        if not self.c.lsync == self.mm.lsync:
            returnVal = False
        if not returnVal:
           msg = QMessageBox()
           msg.setIcon(QMessageBox.Critical)
           msg.setText("NROW, NSAMP, LSYNC must match between cringe and server for tune.\nServer NROW {}, NSAMP {}, LSYNC {}.\nCringe NROW {}, NSAMP {}, LSYNC {}.".format(
           self.c.nrow, self.c.nsamp, self.c.lsync,
           self.mm.seqln, self.mm.NSAMP, self.mm.lsync))
           msg.setWindowTitle("Server Problem")
           msg.setStandardButtons(QMessageBox.Ok)
           msg.exec_()

        return returnVal

    def fullTune(self):
        log.info("start fbb vphi")
        if not self.validateServerSettings():
            return
        self.c.client.setMixToZero()
        fbbtriangle, fbbsigsup, fbbsigsdown = self.FBBvphi()
        fbbstats = vphistats.vPhiStats(fbbtriangle, fbbsigsup)

        # get settings for locked vphis from those vphis
        a2dlockpoints = np.median(fbbstats['midPoint'],axis=1)
        d2aB = np.ones(len(a2dlockpoints))*8000
        I = np.ones(len(a2dlockpoints))*int(round(10*4/float(self.mm.NSAMP))) # choose I for fbb such you get 10 for nsamp = 4, and 5 for nsamp =8
        # set the ARL setting to 10 % more than a fbb vphi
        oldfluxjumpthreshold = self.mm.flux_jump_threshold
        self.mm.setFluxJumpThreshold(np.median(fbbstats["periodXUnits"])*1.1)

        plots = ColPlots(self,self.c.ncol,self.c.nrow)
        plots.plot(fbbtriangle,fbbsigsup)
        plots.title("fbb vphi")
        plots.xlabel("fbb triangle")
        plots.ylabel("error")
        plots.show()

        log.info("start locked fba vphi")
        lfbatriangle, lfbasigsup, lfbasigsdown = self.lockedFBAvphi_colsettings(d2aB, a2dlockpoints, I)
        lfbastats = vphistats.vPhiStats(lfbatriangle, lfbasigsup)



        plots = ColPlots(self,self.c.ncol,self.c.nrow)
        plots.plot(lfbatriangle,lfbasigsup)
        plots.title("locked fba vphi")
        plots.xlabel("fba triangle")
        plots.ylabel("fbb feedback")
        plots.show()


        log.info("taking 2nd fbb vphi, fba set to first minimum from locked vphi")
        d2aA_forfbb2=lfbastats["firstMinimumX"]
        for col in range(self.c.ncol):
            for row in range(self.c.nrow):
                self.mm.setdfbrow(col,row,trib=1,d2aA=d2aA_forfbb2[col,row],data_packet=1)
        tridwell,tristeps,tristepsize=2,9,20
        self.mm.settriangleparams(tridwell,tristeps,tristepsize)
        data = self.c.getNewData(0.1,minimumNumPoints=4096*6)
        fbb = data[0,0,:,1] #triangle
        err = data[0,0,:,0] #signal
        np.save("last_fbb2_vphi",data)
        fbb2triangle, fbb2sigsup, fbb2sigsdown = analysis.conditionvphis(data[:,:,:,1], data[:,:,:,0], tridwell, tristeps, tristepsize)
        fbb2stats = vphistats.vPhiStats(fbb2triangle, fbb2sigsup)



        plots = ColPlots(self,self.c.ncol,self.c.nrow)
        plots.plot(fbb2triangle,fbb2sigsup)
        plots.title("fbb2 vphi,fba set to first minium from locked vphi")
        plots.xlabel("fbb triangle")
        plots.ylabel("error")
        plots.show()


        # the right side of the sq1 curve is at d2aB
        Xup =  fbb2stats["firstMaximumX"] - fbb2stats["firstMinimumX"]
        Xup[Xup<0] += fbb2stats["periodXUnits"][Xup<0]
        Xdown = fbb2stats["firstMinimumX"] - fbb2stats["firstMaximumX"]
        Xdown[Xdown<0] += fbb2stats["periodXUnits"][Xdown<0]
        d2aBupwardSlope = fbb2stats["firstMinimumX"]+lfbastats["modDepth"]+(Xup-lfbastats["modDepth"])/2.0
        d2aBdownwardSlope = fbb2stats["firstMaximumX"]+lfbastats["modDepth"]+(Xdown-lfbastats["modDepth"])/2.0

        if self.lockSlopeSignFBBCheckBox.isChecked():
                d2aB = d2aBupwardSlope
        else:
                d2aB = d2aBdownwardSlope
        for col in range(self.c.ncol):
            for row in range(self.c.nrow):
                period = fbb2stats["periodXUnits"][col,row]
                d2aBval = d2aB[col,row]
                if period > 0 and d2aBval < 0:
                    nPeriodToAdd = int(np.ceil(-d2aBval/period))
                    if nPeriodToAdd > 0:
                        d2aB[col,row]=d2aBval + nPeriodToAdd*period

        oneplot = OnePlot(self)

        plt.plot(fbb2triangle,fbb2sigsup[0,0])
        sq1x = np.linspace(d2aBupwardSlope[0,0],d2aBupwardSlope[0,0]-lfbastats["modDepth"][0,0])
        sq1y = np.interp(sq1x,fbb2triangle,fbb2sigsup[0,0])
        plt.plot(sq1x, sq1y,lw=3,label="upward")
        sq1x = np.linspace(d2aBdownwardSlope[0,0],d2aBdownwardSlope[0,0]-lfbastats["modDepth"][0,0])
        sq1y = np.interp(sq1x,fbb2triangle,fbb2sigsup[0,0])
        plt.plot(sq1x, sq1y,lw=3,label="down")
        plt.plot(fbb2stats["firstMinimumX"][0,0],fbb2stats["firstMinimumY"][0,0],"o",label="minimum")
        plt.plot(fbb2stats["firstMaximumX"][0,0],fbb2stats["firstMaximumY"][0,0],"o",label="maximum")
        if self.lockSlopeSignFBBCheckBox.isChecked():
            plt.title("show where sq1 goes for col 0, row 0\nupward chosen")
        else:
            plt.title("show where sq1 goes for col 0, row 0\ndownward chosen")
        plt.xlabel("fbb triangle")
        plt.ylabel("error")
        plt.legend()
        oneplot.show()
        # oneplot.figure.canvas.draw()
        QApplication.processEvents()  # process gui events




        log.info(("added {} Phi0s".format(self.minimumD2AValueSpin.value())))

        np.save("last_fbb2_firstMinimumX", fbb2stats["firstMinimumX"])
        np.save("last_lfba_modDepth", lfbastats["modDepth"])
        np.save("last_lfba_firstMinimumY", lfbastats["firstMinimumY"])
        np.save("last_fbb2_periodXUnits", fbbstats["periodXUnits"])

        # take fba vphis with correct d2aB values
        tridwell,tristeps,tristepsize=2,8,30
        self.mm.settriangleparams(tridwell,tristeps,tristepsize)
        for col in range(self.c.ncol):
            for row in range(self.c.nrow):
                self.mm.setdfbrow(col,row,tria=1,d2aB=d2aB[col,row])
        data = self.c.getNewData(0.1,minimumNumPoints=4096*6)
        fba = data[0,0,:,1] #triangle
        err = data[0,0,:,0] #signal
        np.save("last_fba_vphi",data)

        fracFromBottom = self.PercentFromBottomSpin.value()*0.01

        fbatriangle, fbasigsup, fbasigsdown = analysis.conditionvphis(data[:,:,:,1], data[:,:,:,0], tridwell, tristeps, tristepsize)
        fbastats = vphistats.vPhiStats(fbatriangle, fbasigsup, fracFromBottom=fracFromBottom)
        with open("last_fbastats","wb") as f:
            pickle.dump(fbastats,f)





        if self.lockSlopeSignCheckBox.isChecked():
            d2aA = fbastats["positiveCrossingFirstX"]
        else:
            d2aA = fbastats["negativeCrossingFirstX"]

        minimum_d2aA = self.minimumD2AValueSpin.value()
        log.info(("using {} as minimum_d2aA".format(self.minimumD2AValueSpin.value())))
        for col in range(self.c.ncol):
            for row in range(self.c.nrow):
                if d2aA[col,row] <= minimum_d2aA and period > 0:
                    nPeriodToAdd = int(np.ceil((minimum_d2aA-d2aA[col,row])/period))
                    d2aA[col,row]+=nPeriodToAdd*fbastats["periodXUnits"][col,row]
                    log.info(("col %g, row %g FBA lockpoint shift up %g phi because it was below %g"%(col, row, nPeriodToAdd, minimum_d2aA)))
        a2d = fbastats["crossingPoint"]

        fbatriangles = np.zeros((self.c.ncol, self.c.nrow, len(fbatriangle)),dtype="int64")
        fbasigsupShifted = fbasigsup[:]
        for col in range(self.c.ncol):
            for row in range(self.c.nrow):
                fbatriangles[col,row,:]=fbatriangle-d2aA[col,row]
                fbasigsupShifted[col,row,:]=fbasigsup[col,row,:]-a2d[col,row]


        plots = ColPlots(self,self.c.ncol,self.c.nrow)
        plots.plotbigx(fbatriangles,fbasigsupShifted)
        plots.title("final fba vphis analyzed and shifted to lockpoint")
        plots.xlabel("fba triangle")
        plots.ylabel("error")
        plots.show()


        ISlopeProduct = self.ISlopeProductSpin.value()
        MixSlopeProduct = self.MixSlopeProductSpin.value()

        if self.lockSlopeSignCheckBox.isChecked():
            I = ISlopeProduct/fbastats["positiveCrossingSlope"]
        else:
            I = ISlopeProduct/fbastats["negativeCrossingSlope"]

        I=np.array(np.round(I),dtype="int64")
        chanisgood = np.ones(I.shape,dtype="bool")
        chanisgood[I>511]=False
        chanisgood[I<-511]=False
        chanisgood[fbastats["periodXUnits"]<200]=False

        sq1periods = np.median(fbastats["periodXUnits"],axis=1)

        if self.lockSlopeSignCheckBox.isChecked():
            Mix = chanisgood*MixSlopeProduct/fbastats["positiveCrossingSlope"]
        else:
            Mix = chanisgood*MixSlopeProduct/fbastats["negativeCrossingSlope"]


        np.save("last_mix_values", Mix/100.0)
        writeMixFile(os.path.expanduser("~/nasa_daq/matter/autotune_mix.mixing_config"), Mix/100.0)
        # cringe should have created this directory earlier
        np.save(os.path.expanduser("~/.cringe/mix_fractions"), Mix/100.0)

        for col in range(self.c.ncol):
            for row in range(self.c.nrow):
                if chanisgood[col,row]:
                    self.mm.setdfbrow(col,row,a2d=a2d[col,row],I=I[col,row],d2aA=d2aA[col,row],d2aB=d2aB[col,row],FBA=1, ARL=1)
                else:
                    # set D2A values for bad channels to the midpoint of the intended distribution of other D2A values
                    # the minimum value is minimum_d2aA, all D2A values should be with sq1periods[col] of the minimum
                    # this should reduce the maximum difference in D2A between two different rows
                    # having large D2A changes between rows causes crosstalk
                    self.mm.setdfbrow(col,row,a2d=0,I=0,d2aA=minimum_d2aA+0.5*sq1periods[col],d2aB=0,FBA=0, ARL=0)

        sq1periodsstrs = ["col %g %g"%(i, int(round(sq1periods[i]))) for i in range(len(sq1periods))]
        log.info(("median sq1 periods by column (arbs):\n "+"\n".join(sq1periodsstrs)))
        log.info(("median of all columns: %g arbs"%np.median(sq1periods)))
        goodfluxjumpthreshold=int(np.median(sq1periods)/2)
        log.info(("typically the ARL FluxJumpThreshold should be around 1/2 the sq1 period, so %g could be good"%goodfluxjumpthreshold))
        log.info(("This value was set for you, your previous value was %g"%oldfluxjumpthreshold))
        self.mm.setFluxJumpThreshold(goodfluxjumpthreshold)

        if self.shouldSendMixAfterFullTune():
            log.info("sending mix values after full tune")
            log.info(Mix)
            Mix[np.isnan(Mix)]=0 # don't send NaN, its invalid for mix
            log.info(Mix)
            log.info((Mix/100.0))
            self.c.client.setMix(Mix/100.0)

    def prune_bad_channels(self):
        log.info("prune_bad_channels")
        min_amplitude = 600
        max_noise_std = 1
        with open("last_fbastats","r") as f:
            vphistats = pickle.load(f)
        # log.info([k for k in vphistats.keys()])
        log.info((vphistats["modDepth"]))
        assert(vphistats["modDepth"].shape == (self.c.ncol, self.c.nrow))
        for col in range(self.c.ncol):
            for row in range(self.c.nrow):
                errorChan = col*2*self.c.nrow+row*2
                fbChan = errorChan+1
                if vphistats["modDepth"][col,row] < min_amplitude:

                # turn off fb on row
                    self.mm.setdfbrow(col,row) # d2aA and d2aB should match next row? unless that row is bad?
                    self.c.client.setMixChannel(fbChan,0)
        time.sleep(1)
        # do the loop twice, so you can actually see this output despite all the
        # crap cringe prints
        for col in range(self.c.ncol):
            for row in range(self.c.nrow):
                errorChan = col*2*self.c.nrow+row*2
                fbChan = errorChan+1
                if vphistats["modDepth"][col,row] < min_amplitude:
                    log.info(("c%gr%g chan %g has amplitdue %0.2f, less than min=%0.f, turning off feedback and mix"%(
                        col,row, fbChan, vphistats["modDepth"][col,row], min_amplitude
                    )))        # data = self.c.getNewData(0.1,minimumNumPoints=4096*6)
        # fba = data[0,0,:,1] #triangle
        # err = data[0,0,:,0] #signal
        # fba_std = np.std(data[:,:,:,1],axis=2)
        # err_std = np.std(data[:,:,:,0],axis=2)
        # log.info(fba_std)
        # log.info(err_std)



    def grab_and_set_d2aA_values(self):
        # assume feedback is on

        data = self.c.getNewData(0.1,minimumNumPoints=4096)
        fba = data[0,0,:,1] # feedback signal, should be locked
        err = data[0,0,:,0] # signal, should be near zero

        d2aA = np.median(data[:,:,:,1],axis=-1)

        for col in range(self.c.ncol):
            for row in range(self.c.nrow):
                self.mm.changedfbrow(col,row, d2aA=d2aA[col,row])

    def plotnoise(self):
        """dont change anything, grab some data, take the fft, plot the noise"""
        num_points = 2**14
        navgs = 40
        sample_spacing_s = 1/float(self.c.sample_rate)
        log.info("taking noise psd, not changin any settings, so make sure you have FBA locked and the correct send mode, and the mix on")
        log.info("sample time = %g s"%sample_spacing_s)
        freqs = np.fft.rfftfreq(num_points, sample_spacing_s)
        ffts = np.zeros((self.c.ncol, self.c.nrow, len(freqs)))
        df = float(freqs[1]-freqs[0])
        log.info("frequency spacing of psd =  %0.2f hz"%df)

        for i in range(navgs):
            data = self.c.getNewData(0.001,minimumNumPoints=num_points, exactNumPoints=True)
            for col in range(self.c.ncol):
                for row in range(self.c.nrow):
                    ffts[col,row,:] += np.abs(np.fft.rfft(data[col,row,:,1]))
        ffts/=float(navgs)
        psd = 2*ffts/np.sqrt(float(self.c.sample_rate)*num_points)


        plots = ColPlots(self,self.c.ncol,self.c.nrow)
        plots.semilogx(freqs[1:], psd[:,:,1:])
        plots.title("FBA fft")
        plots.xlabel("frequency (hz)")
        plots.ylabel("arbs/sqrt(hz)")
        plots.xlim((freqs[1],freqs[-1]))
        plots.ylim((0,0.08))
        plots.show()

        np.save("last_noise_freqs_hz", freqs)
        np.save("last_noise_psd_arbs_per_sqrt_hz", ffts)

        flo = 15000
        fhi = 40000
        ilo,ihi = np.searchsorted(freqs,[flo,fhi])
        medpsd = np.median(psd[:,:,ilo:ihi],axis=-1)
        medpsdcol = np.median(medpsd,axis=-1)
        noisestr=["col %d %0.4f"%(i, medpsdcol[i]) for i in range(self.c.ncol)]
        log.info("median noise arbs/sqrt(hz) from %0.2f-%0.2f hz:\n"%(freqs[ilo],freqs[ihi])+"\n".join(noisestr))
        log.info(("median across all columns: %g arbs/sqrt(hz)"%np.median(medpsdcol)))
        log.info("saved files last_noise_freqs_hz and last_noise_psd_arbs_per_sqrt_hz, copy or rename if you want to keep them")

class SettlingTimeSweeper(QWidget):
    def __init__(self, parent, mm, client):
        super(type(self), self).__init__(parent)
        self.layout = QVBoxLayout(self)
        self.mm=mm
        self.c=client

        self.layout.addWidget(QLabel("Settling time sweeps. Run full tune first. Settings only apply to locked."))


        settsweepbutton= QPushButton(self,text="Sweep unlocked")
        settsweepbutton.clicked.connect(self.sweepUnlocked)
        self.layout.addWidget(settsweepbutton)

        layout = QHBoxLayout()
        self.adcLoSpin = QSpinBox()
        self.adcLoSpin.setRange(1,99)
        self.adcLoSpin.setValue(35)
        layout.addWidget(QLabel("% from bottom of FBB Vphi lo"))
        layout.addWidget(self.adcLoSpin)
        self.layout.addLayout(layout)

        layout = QHBoxLayout()
        self.adcHiSpin = QSpinBox()
        self.adcHiSpin.setRange(1,99)
        self.adcHiSpin.setValue(65)
        layout.addWidget(QLabel("% from bottom of FBB Vphi hi"))
        layout.addWidget(self.adcHiSpin)
        self.layout.addLayout(layout)

        lockedsweepbutton= QPushButton(self,text="Sweep locked")
        lockedsweepbutton.clicked.connect(self.sweepLocked)
        self.layout.addWidget(lockedsweepbutton)


    def FBBvphi_from_file(self):
        tridwell,tristeps,tristepsize=2,9,20

        try:
            data = np.load("last_fbb_vphi.npy")
        except:
            log.info("file last_fbb_vphi.npy doesn't exist, try running a full tune first")

        outtriangle, outsigsup, outsigsdown = analysis.conditionvphis(data[:,:,:,1], data[:,:,:,0], tridwell, tristeps, tristepsize)
        return outtriangle, outsigsup, outsigsdown

    def sweepUnlocked(self):
        """set nsamp to 1, sweep thru settling times, measure error and fb,
        plot them, change back nsamp and settinling time to original values"""
        log.info("starting sweep Unlocked")
        self.c.client.setMixToZero()

        log.info("taking FBB vphi to find good adc values")
        adc_lo_frac = self.adcLoSpin.value()*0.01
        adc_hi_frac = self.adcHiSpin.value()*0.01
        fbbtriangle, fbbsigsup, fbbsigsdown = self.FBBvphi_from_file()
        fbbstats = vphistats.vPhiStats(fbbtriangle, fbbsigsup)

        adc_lo = np.zeros(self.c.ncol,dtype="int64")
        adc_hi = np.zeros(self.c.ncol,dtype="int64")
        for col in range(self.c.ncol):
            modDepth = np.median(fbbstats["modDepth"][col,:])
            midPoint = np.median(fbbstats["midPoint"][col,:])
            bottom = midPoint-modDepth*0.5
            adc_lo[col] = bottom+modDepth*adc_lo_frac
            adc_hi[col] = bottom+modDepth*adc_hi_frac

        log.info("locking at adc values to find d2aA values")
        # lock at desired a2d points
        for col in range(self.c.ncol):
            for row in range(self.c.nrow):
                if row%2 == 0:
                    a2d = adc_lo[col]
                else:
                    a2d = adc_hi[col]
                self.mm.changedfbrow(col,row,a2d=a2d,d2aA=8000,tria=0,trib=0,FBA=1,FBB=0,ARL=0,data_packet=0,dynamic=1)
        log.info("done locking at adc values to find d2aA values")


        # grab data to get d2aA values
        data = self.c.getNewData(0.1,minimumNumPoints=4096)
        fba = data[0,0,:,1] # feedback signal, should be locked
        err = data[0,0,:,0] # signal, should be near zero
        d2aA = np.median(data[:,:,:,1],axis=-1)
        log.info(d2aA)


        log.info("setting d2aA values")
        for col in range(self.c.ncol):
            for row in range(self.c.nrow):
                self.mm.changedfbrow(col,row, a2d=0,d2aA=d2aA[col,row], FBA=0)
        log.info("done setting d2aA values")



        oldSETT = self.mm.SETT
        oldNSAMP = self.mm.NSAMP
        self.mm.setnsamp(1)
        maxSETT = self.mm.lsync-2-1

        SETTs = np.arange(maxSETT,-1,-1)
        errs = np.zeros((self.c.ncol,self.c.nrow, len(SETTs)))
        fbs = np.zeros((self.c.ncol,self.c.nrow, len(SETTs)))
        errstds = np.zeros((self.c.ncol,self.c.nrow, len(SETTs)))
        fbstds = np.zeros((self.c.ncol,self.c.nrow, len(SETTs)))
        for i,SETT in enumerate(SETTs):
            log.info(("sweep SETT = %g"%SETT))
            self.mm.setSETT(SETT)

            data = self.c.getNewData(0.01,minimumNumPoints=10000,divideNsamp=False)
            err = np.median(data[:,:,:,0],axis=2)
            errstd = np.std(data[:,:,:,0],axis=2)
            fb  = np.median(data[:,:,:,1],axis=2)
            fbstd  = np.std(data[:,:,:,1],axis=2)
            errs[:,:,i]=err
            errstds[:,:,i]=errstd
            fbs[:,:,i]=fb
            fbstds[:,:,i]=fbstd

        np.save("last_unlocked_sett_sweep_SETTs",SETTs)
        np.save("last_unlocked_sett_sweep_errs",errs)

        plots = ColPlots(self,self.c.ncol,self.c.nrow)
        plots.plot(SETTs,errstds)
        plots.title("sett sweep, prop delay %g, dfb delay %g, bad delay %g"%(self.mm.prop_delay, self.mm.dfb_delay, self.mm.bad_delay))
        plots.xlabel("SETT")
        plots.ylabel("error std dev")
        plots.show()

        plots = ColPlots(self,self.c.ncol,self.c.nrow)
        plots.plot(SETTs,errs)
        plots.title("sett sweep, prop delay %g, dfb delay %g, bad delay %g"%(self.mm.prop_delay, self.mm.dfb_delay, self.mm.bad_delay))
        plots.xlabel("SETT")
        plots.ylabel("error")
        plots.show()

        self.mm.setnsamp(oldNSAMP)
        self.mm.setSETT(oldSETT)

    def sweepLocked(self):

        oldSETT = self.mm.SETT
        oldNSAMP = self.mm.NSAMP
        self.mm.setnsamp(1)
        maxSETT = self.mm.lsync-2-1

        SETTs = np.arange(maxSETT,-1,-1)
        errs = np.zeros((self.c.ncol,self.c.nrow, len(SETTs)))
        fbs = np.zeros((self.c.ncol,self.c.nrow, len(SETTs)))
        errstds = np.zeros((self.c.ncol,self.c.nrow, len(SETTs)))
        fbstds = np.zeros((self.c.ncol,self.c.nrow, len(SETTs)))
        for i,SETT in enumerate(SETTs):
            log.info(("sweep SETT = %g"%SETT))
            self.mm.setSETT(SETT)

            data = self.c.getNewData(0.01,minimumNumPoints=10000,divideNsamp=False)
            err = np.median(data[:,:,:,0],axis=2)
            errstd = np.std(data[:,:,:,0],axis=2)
            fb  = np.median(data[:,:,:,1],axis=2)
            fbstd  = np.std(data[:,:,:,1],axis=2)
            errs[:,:,i]=err
            errstds[:,:,i]=errstd
            fbs[:,:,i]=fb
            fbstds[:,:,i]=fbstd


        plots = ColPlots(self,self.c.ncol,self.c.nrow)
        plots.plot(SETTs,fbstds)
        plots.title("sett sweep, prop delay %g, dfb delay %g, bad delay %g"%(self.mm.prop_delay, self.mm.dfb_delay, self.mm.bad_delay))
        plots.xlabel("SETT")
        plots.ylabel("fbA std dev")
        plots.show()

        plots = ColPlots(self,self.c.ncol,self.c.nrow)
        plots.plot(SETTs,fbs)
        plots.title("sett sweep, prop delay %g, dfb delay %g, bad delay %g"%(self.mm.prop_delay, self.mm.dfb_delay, self.mm.bad_delay))
        plots.xlabel("SETT")
        plots.ylabel("fbA")
        plots.show()

        self.mm.setnsamp(oldNSAMP)
        self.mm.setSETT(oldSETT)


class BitErrorDemo(QWidget):
    def __init__(self, parent, mm, client):
        super(type(self), self).__init__(parent)
        self.layout = QHBoxLayout(self)
        self.mm=mm
        self.c=client

        self.layout.addWidget(QLabel("Bit Error Test"))

        self.startbutton = QPushButton(self,text="start")
        self.startbutton.clicked.connect(self.start)
        self.layout.addWidget(self.startbutton)

        self.stopbutton = QPushButton(self,text="stop")
        self.stopbutton.clicked.connect(self.stop)
        self.layout.addWidget(self.stopbutton)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.timeouthandler)

        self.ehhw = 25 # error histogram halfwidth


    def timeouthandler(self):
        data = self.c.getNewData(0.001,minimumNumPoints=self.c.sample_rate*1.0)

        for col in range(self.c.ncol):
            for row in range(self.c.nrow):
                fbdiff = np.abs(np.diff(data[col,row,:,1]))
                #fb diff should contain only 0s and tristepsize and -tristepsize as values, any other value is an error
                self.ntotal[col]+=len(fbdiff)
                self.nwrong[col]+=len(fbdiff)-np.sum(np.logical_or(fbdiff == 0,fbdiff==self.tristepsize))
                bins = np.arange(self.errormedians[col,row]-self.ehhw,self.errormedians[col,row]+self.ehhw)
                c, _ = np.histogram(data[col,row,:,0],bins)
                self.hists[col,row,:]+=c
        self.plots.cla()
        self.plots.xlabel("error - median(error)")
        self.plots.ylabel("number of occurences")
        self.plots.plottitle([": %0.2g/%0.2g"%(self.nwrong[col],self.ntotal[col]) for col in range(self.c.ncol)])
        self.plots.semilogy(np.arange(-self.ehhw,self.ehhw-1),self.hists)
        self.showplots()


    def makeplots(self):
        self.plots = ColPlots(self,self.c.ncol,self.c.nrow)
        self.plots.title("To stop: move this window and click stop in Cringe/Tune, then close this window.\nTriangle on FBA, measuring error signal. Histograms of error signal - median error signal.\nNumber of sample errors shown in each plot title. Sample error = 1 or more bit wrong in a single triangle value.")


    def start(self):
        self.makeplots()
        self.c.startclient()
        self.tridwell,self.tristeps,self.tristepsize=2,8,10
        self.mm.settriangleparams(self.tridwell,self.tristeps,self.tristepsize)
        self.mm.setdfball(tria=1)
        data = self.c.getNewData(0.001)
        self.errormedians = np.zeros((self.c.ncol,self.c.nrow),dtype="int64")
        self.errorstds = np.zeros((self.c.ncol,self.c.nrow))
        self.nwrong = np.zeros(self.c.ncol,dtype="int64")
        self.ntotal = np.zeros(self.c.ncol,dtype="int64")
        self.hists = np.zeros((self.c.ncol,self.c.nrow,self.ehhw*2-1),dtype="int64")
        for col in range(self.c.ncol):
            for row in range(self.c.nrow):
                self.errormedians[col,row] = np.median(data[col,row,:,0])
                self.errorstds[col,row] = np.std(data[col,row,:,0])
        self.timer.start(250)
        self.show()

    def stop(self):
        self.timer.stop()

    def oneOffVphi(self):
        tridwell,tristeps,tristepsize=2,8,10
        self.mm.settriangleparams(tridwell,tristeps,tristepsize)
        self.mm.setdfball(tria=1)
        data = self.c.getNewData(0.1)

        nframes = data.shape[-2]

        nwrong = np.zeros((self.c.ncol,self.c.nrow))
        nwrongcol = np.zeros(self.c.ncol)
        hists = np.zeros((self.c.ncol,self.c.nrow,199))
        binsc = []
        for col in range(self.c.ncol):
            for row in range(self.c.nrow):
                fbdiff = np.abs(np.diff(data[col,row,:,1]))
                #fb diff should contain only 0s and tristepsize and -tristepsize as values, any other value is an error
                nwrong[col,row]=len(fbdiff)-np.sum(np.logical_or(fbdiff == 0,fbdiff==tristepsize))

                med = np.median(data[col,row,:,0])
                std = np.std(data[col,row,:,0])
                bins = np.arange(med-100,med+100)
                c, _ = np.histogram(data[col,row,:,0],bins)
                hists[col,row,:] = np.log(c)

            nwrongcol[col] = np.sum(nwrong[col,:])
            log.info(("col %g, %g/%g fb values wrong"%(col, nwrongcol[col],nframes*self.c.nrow)))

        self.makeplots()
        self.plots.plot(np.arange(-100,99),hists)
        self.plots.show()

        fb = data[0,0,:,1]
        err = data[0,0,:,0]

        plots = ColPlots(self,self.c.ncol,self.c.nrow)
        plots.show()
        plots.plot(fb,data[:,:,:,0])

    def showplots(self):
        self.plots.show()

class BiasSweeper(QWidget):
    def __init__(self, parent, mm, client):
        super(type(self), self).__init__(parent)
        self.layout = QHBoxLayout(self)
        self.mm=mm
        self.c=client

        self.layout.addWidget(QLabel("Squid bias sweeper."))

        sweepbutton= QPushButton(self,text="Sweep BAD16 high, locked FBA vphis.")
        sweepbutton.clicked.connect(self.sweepBAD16FBA)
        self.layout.addWidget(sweepbutton)

        sweepbutton= QPushButton(self,text="Sweep SAb bias, FBB vphis.")
        sweepbutton.clicked.connect(self.sweepSABFBB)
        self.layout.addWidget(sweepbutton)

        sweepbutton= QPushButton(self,text="Sweep SQ1b bias, locked FBA vphis.")
        sweepbutton.clicked.connect(self.sweepSQ1FBA)
        self.layout.addWidget(sweepbutton)

    def FBBvphi_from_file(self):
        tridwell,tristeps,tristepsize=2,9,20

        try:
            data = np.load("last_fbb_vphi.npy")
        except:
            log.info("file last_fbb_vphi.npy doesn't exist, try running a full tune first")

        outtriangle, outsigsup, outsigsdown = analysis.conditionvphis(data[:,:,:,1], data[:,:,:,0], tridwell, tristeps, tristepsize)
        return outtriangle, outsigsup, outsigsdown

    def lockedFBAvphi_colsettings(self, d2aB, a2d, I):
        tridwell,tristeps,tristepsize=2,9,15
        self.mm.settriangleparams(tridwell,tristeps,tristepsize)

        for col in range(self.c.ncol):
            self.mm.setdfballrow(col,tria=1,ARL=1,data_packet=2, FBB=1, I=I[col], d2aB=d2aB[col], a2d=a2d[col])


        data = self.c.getNewData(0.1,minimumNumPoints=4096*6,sendMode=2)
        fba = data[0,0,:,1] #signal
        fbb = data[0,0,:,0] #triangle

        outtriangle, outsigsup, outsigsdown = analysis.conditionvphis(data[:,:,:,0], data[:,:,:,1], tridwell, tristeps, tristepsize)
        return outtriangle, outsigsup, outsigsdown

    def sweepBAD16FBA(self):


        self.c.client.setMixToZero()
        fbbtriangle, fbbsigsup, fbbsigsdown = self.FBBvphi_from_file()
        fbbstats = vphistats.vPhiStats(fbbtriangle, fbbsigsup)

        # get settings for locked vphis from those vphis
        a2dlockpoints = np.median(fbbstats['midPoint'],axis=1)
        d2aB = np.ones(len(a2dlockpoints))*8000
        I = np.ones(len(a2dlockpoints))*int(round(10*4/float(self.mm.NSAMP))) # choose I for fbb such you get 10 for nsamp = 4, and 5 for nsamp =8
        # set the ARL setting to 10 % more than a fbb vphi
        oldfluxjumpthreshold = self.mm.flux_jump_threshold
        self.mm.setFluxJumpThreshold(np.median(fbbstats["periodXUnits"])*1.1)



        vals = np.arange(0,15000,1000)
        modDepths = np.zeros((self.c.ncol, self.c.nrow, len(vals)))


        for i in range(len(vals)):
            val = vals[i]
            self.mm.setbaddacHighsSame(val)
            lfbatriangle, lfbasigsup, lfbasigsdown = self.lockedFBAvphi_colsettings(d2aB, a2dlockpoints, I)
            lfbastats = vphistats.vPhiStats(lfbatriangle, lfbasigsup)

            modDepths[:,:,i] = lfbastats["modDepth"]

        self.mm.setFluxJumpThreshold(oldfluxjumpthreshold)

        plots = ColPlots(self,self.c.ncol,self.c.nrow)
        plots.plot(vals,modDepths)
        plots.title("FBA vphis, sweep BAD16 high")
        plots.xlabel("BAD16 high")
        plots.ylabel("FBA modDepth")
        plots.show()

    def sweepSABFBB(self):
        tridwell,tristeps,tristepsize=2,9,20
        self.mm.settriangleparams(tridwell,tristeps,tristepsize)
        self.mm.setdfball(trib=1,data_packet=1) # FBB, ERR

        vals = np.arange(0,40000,5000)
        self.setSAB(vals[0])
        time.sleep(1) # go to the first value, and wait a while
        # this avoid settling time issues from the potentially large change

        # setup for vphi, ignore first datadata
        # probably there are command left in serial que after big cahnges
        data = self.c.getNewData(0.1,minimumNumPoints=4096*6)
        fbbtriangle, fbbsigsup, fbbsigsdown = analysis.conditionvphis(data[:,:,:,1], data[:,:,:,0], tridwell, tristeps, tristepsize)


        modDepths = np.zeros((self.c.ncol, self.c.nrow, len(vals)))
        for i in range(len(vals)):
            val = vals[i]
            self.setSAB(val)
            data = self.c.getNewData(0.1,minimumNumPoints=4096*6)
            fbbtriangle, fbbsigsup, fbbsigsdown = analysis.conditionvphis(data[:,:,:,1], data[:,:,:,0], tridwell, tristeps, tristepsize)
            fbbstats = vphistats.vPhiStats(fbbtriangle, fbbsigsup)
            modDepths[:,:,i] = fbbstats["modDepth"]


        plots = ColPlots(self,self.c.ncol,1)
        plots.plot(vals,modDepths)
        plots.title("FBA vphis, sweep SAb")
        plots.xlabel("SAb")
        plots.ylabel("FBB modDepth")
        plots.show()

    def sweepSQ1FBA(self):
        self.c.client.setMixToZero()
        fbbtriangle, fbbsigsup, fbbsigsdown = self.FBBvphi_from_file()
        fbbstats = vphistats.vPhiStats(fbbtriangle, fbbsigsup)

        # get settings for locked vphis from those vphis
        a2dlockpoints = np.median(fbbstats['midPoint'],axis=1)
        d2aB = np.ones(len(a2dlockpoints))*8000
        I = np.ones(len(a2dlockpoints))*int(round(10*4/float(self.mm.NSAMP))) # choose I for fbb such you get 10 for nsamp = 4, and 5 for nsamp =8
        # set the ARL setting to 10 % more than a fbb vphi
        oldfluxjumpthreshold = self.mm.flux_jump_threshold
        self.mm.setFluxJumpThreshold(np.median(fbbstats["periodXUnits"])*1.1)

        vals = np.arange(0,8000,600)
        self.setSQ1(vals[0])
        time.sleep(.2) # go to the first value, and wait a while
        # this avoid settling time issues from the potentially large change

        modDepths = np.zeros((self.c.ncol, self.c.nrow, len(vals)))
        for i in range(len(vals)):
            val = vals[i]
            self.setSQ1(val)
            data = self.c.getNewData(0.1,minimumNumPoints=4096*2)
            lfbatriangle, lfbasigsup, lfbasigsdown = self.lockedFBAvphi_colsettings(d2aB, a2dlockpoints, I)
            lfbastats = vphistats.vPhiStats(lfbatriangle, lfbasigsup)
            modDepths[:,:,i] = lfbastats["modDepth"]

#             plots = ColPlots(self,self.c.ncol,self.c.nrow)
#             plots.plot(lfbatriangle,lfbasigsup)
#             plots.title("locked fba vphis vs SQ1b, val =%g"%val)
#             plots.xlabel("fba")
#             plots.ylabel("fbb (locked signal)")
#             plots.show()
        self.mm.setFluxJumpThreshold(oldfluxjumpthreshold)

        plots = ColPlots(self,self.c.ncol,self.c.nrow)
        plots.plot(vals,modDepths)
        plots.title("FBA vphis, sweep SQ1")
        plots.xlabel("SQ1")
        plots.ylabel("FBA modDepth")
        plots.show()

    def setSAB(self, val):
        towercard = self.mm.cringe.tower_widget.towercards["SAb"]
        for towerchannel in towercard.towerchannels:
            towerchannel.dacspin.setValue(val)

    def setSQ1(self, val):
        towercard = self.mm.cringe.tower_widget.towercards["SQ1b"]
        for towerchannel in towercard.towerchannels:
            towerchannel.dacspin.setValue(val)

class OnePlot(QDialog):
    def __init__(self, parent):
        super(type(self), self).__init__(parent)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas,self)
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)
        self.ax = plt.gca()


class ColPlots(QDialog):
    def __init__(self, parent,ncol,nrow):
        super(type(self), self).__init__(parent)
        self.ncol = ncol
        self.nrow = nrow
        self.numXSubplots = int((ncol+1)/2.0)
        self.numYSubplots = 1+int(ncol>1)
        self.figure = plt.figure(figsize=(self.numXSubplots*6, self.numYSubplots*4))
        self.canvas = FigureCanvas(self.figure)
        self.titlelabel = QLabel("")
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout = QVBoxLayout()
        layout.addWidget(self.titlelabel)
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)

        self.createaxes()
        self.plottitle()
        self.xlabel("triangle")
        self.ylabel("err")
        self.title("joint title")

        cid = self.canvas.mpl_connect('pick_event', self.onclick)

    def onclick(self, event):
        log.info(event.artist.get_label()+" was clicked")

    def createaxes(self):
        self.axes = []
        for col in range(self.ncol):
            ax = plt.subplot(self.numYSubplots, self.numXSubplots, col+1)
            self.axes.append(ax)

    def xlabel(self,s):
        for col in range(self.ncol):
            ax=self.axes[col]
            if col>=self.numXSubplots*(self.numYSubplots-1):
                ax.set_xlabel(s)

    def ylabel(self,s):
        for col in range(self.ncol):
            ax=self.axes[col]
            if col%self.numXSubplots==0:
                ax.set_ylabel(s)

    def plottitle(self,s=[""]*32):
        for col in range(self.ncol):
            ax=self.axes[col]
            ax.set_title(("col %g"%col)+s[col])

    def plot(self,x,y):
        for col in range(self.ncol):
            ax=self.axes[col]
            for row in range(self.nrow):
                ax.plot(x,y[col,row,:],".-", picker=5, label="col %g, row%g"%(col,row))
        self.draw()

    def plotbigx(self, x, y):
        for col in range(self.ncol):
            ax=self.axes[col]
            for row in range(self.nrow):
                ax.plot(x[col,row,:],y[col,row,:], picker=5, label="col %g, row%g"%(col,row))
        self.draw()

    def semilogy(self,x,y):
        for col in range(self.ncol):
            ax=self.axes[col]
            for row in range(self.nrow):
                ax.semilogy(x,y[col,row,:],".", picker=5, label="col %g, row%g"%(col,row))
        self.draw()

    def semilogx(self,x,y):
        for col in range(self.ncol):
            ax=self.axes[col]
            for row in range(self.nrow):
                ax.semilogx(x,y[col,row,:], picker=5, label="col %g, row%g"%(col,row))
        self.draw()


    def plotdiffx(self,x,y):
        for col in range(self.ncol):
            ax=self.axes[col]
            for row in range(self.nrow):
                ax.plot(x[col],y[col,row,:],".", picker=5, label="col %g, row%g"%(col,row))
        self.draw()



    def title(self,s=""):
        self.titlelabel.setText(s)

    def xlim(self,xlims):
        for col in range(self.ncol):
            ax=self.axes[col]
            ax.set_xlim(xlims)

    def ylim(self,ylims):
        for col in range(self.ncol):
            ax=self.axes[col]
            ax.set_ylim(ylims)

    def cla(self):
        for col in range(self.ncol):
            ax=self.axes[col]
            ax.clear()

    def draw(self):
        self.canvas.draw()
        QApplication.processEvents()  # process gui events


def writeMixFile(fname, mix):
    log.info(('writing mix to %s'%fname))
    ncol, nrow = mix.shape
    f = open(fname, 'w')
    for col in range(ncol):
        for row in range(nrow):
            errorChan = col*2*nrow+row*2
            fbChan = errorChan+1
            f.write('CH%d_decimateAvgFlag: 1\nCH%d_decimateFlag: 0\nCH%d_mixFlag: 1\nCH%d_mixInversionFlag: 0\n'%(errorChan, errorChan, errorChan, errorChan))
            f.write('CH%d_decimateAvgFlag: 1\n'%fbChan)
            f.write('CH%d_decimateFlag: 0\n'%fbChan)
            f.write('CH%d_mixFlag: %d\n'%(fbChan,1)) # 0 disables mix
            f.write('CH%d_mixInversionFlag: 0\n'%fbChan)
            f.write('CH%d_mixLevel: %f\n'%(fbChan, mix[col,row]))
    f.close()
