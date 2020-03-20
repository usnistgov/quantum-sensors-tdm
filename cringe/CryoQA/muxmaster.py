import numpy as np

class MuxMaster():
    def __init__(self,cringe):
        dfbraps = []
        badraps = []
        badstates = []
        for idx, val in enumerate(cringe.class_vector):
            #if val == "DFBCLK":
                #print("DFBCLK, dont care")
            if val == "DFBx2":
                #print(idx, val)
                dfbraps.append(cringe.crate_widgets[idx].dfbx2_widget1)
                dfbraps.append(cringe.crate_widgets[idx].dfbx2_widget2)
            if val == "BAD16":
                #print(idx,val)
                badraps.append(cringe.crate_widgets[idx].badrap_widget1)
                badstates.append(cringe.crate_widgets[idx].badrap_widget2)
        #print(dfbraps)
        #print(badraps)
        #print(badstates)
        self.dfbraps = dfbraps
        self.badraps = badraps
        self.badstates = badstates
        self.cringe=cringe

    def getdacAoffsets(self):
        dacAoffsets = []
        for dfbrap in self.dfbraps:
            for dfbchn in dfbrap.state_vectors:
                dacAoffsets.append(dfbchn.d2a_A_spin.value())
        print(dacAoffsets)

    def getdacBoffsets(self):
        dacBoffsets = []
        for dfbrap in self.dfbraps:
            for dfbchn in dfbrap.state_vectors:
                dacBoffsets.append(dfbchn.d2a_B_spin.value())
        print(dacBoffsets)

    def getadcLockpoints(self):
        adcLockpoints = []
        for dfbrap in self.dfbraps:
            for dfbchn in dfbrap.state_vectors:
                adcLockpoints.append(dfbchn.a2d_lockpt_spin.value())
        return adcLockpoints

    def getbaddacHighs(self):
        print((self.getbadroworder()))
        baddacHighs = []
        print((len(self.badraps), len(self.badraps[0].chn_vectors)))
        for badrap in self.badraps:
            for chn in badrap.chn_vectors:

                baddacHighs.append(chn.d2a_hi_slider.value())
        return baddacHighs

    def setbaddacHighsSame(self, val):
        for badrap in self.badraps:
            for chn in badrap.chn_vectors:
                chn.d2a_hi_slider.setValue(val)


    def getbadroworder(self):
        roworder = []
        for i,sv_array in enumerate(self.badstates):
            print("a")
            for sv in sv_array.state_vectors[:self.seqln]:
                checked = [button.isChecked() for button in sv.buttons]
                assert(np.sum(checked)==1)
                roworder.append(i*self.seqln+find(checked)[0])
                print(checked)
        return roworder

    def changedfbrow(self,col=None,row=None,tria=None,trib=None,a2d=None,d2aA=None,d2aB=None,P=None,I=None,FBA=None,FBB=None,ARL=None,data_packet=None,dynamic=None):
        """
        setdfbrow(self,col=0,row=0,tria=0,trib=0,a2d=0,d2aA=0,d2aB=0,P=0,I=0,FBA=0,FBB=0,ARL=0,data_packet=0,dynamic=1)
        set every possibly setting for a single dfbx2 row, all values have defaults of 0 or off, or FBA,ERR.
        dynamic has default on so that everything gets sent.
        goes through the gui so everything stays in sync
        """
        dfbrap=self.dfbraps[col]
        if row == "master":
            chn = dfbrap.master_vector
        else:
            chn = dfbrap.state_vectors[row]
        if dynamic is not None:
            chn.lock_button.setChecked(dynamic) #this is the dynamic button, when true all commands are send when gui is changed, the send all button should not be needed, do this first so I don't need to send manually
        if a2d is not None:
            chn.a2d_lockpt_spin.setValue(a2d)
        if d2aA is not None:
            chn.d2a_A_spin.setValue(d2aA)
        if d2aB is not None:
            chn.d2a_B_spin.setValue(d2aB)
        if P is not None:
            chn.P_spin.setValue(P)
        if I is not None:
            chn.I_spin.setValue(I)
        if data_packet is not None:
            chn.data_packet.setCurrentIndex(data_packet)
        if tria is not None:
            chn.TriA_button.setChecked(tria)
        if trib is not None:
            chn.TriB_button.setChecked(trib)
        if ARL is not None:
            chn.ARL_button.setChecked(ARL)
        if FBA is not None:
            chn.FBA_button.setChecked(FBA)
        if FBB is not None:
            chn.FBB_button.setChecked(FBB)

    def lockAall(self, lock=True):
        for col in range(len(self.dfbraps)):
            for row in range(self.seqln):
                dfbrap=self.dfbraps[col]
                chn = dfbrap.state_vectors[row]
                chn.FBA_button.setChecked(lock)


    def setdfbrow(self,col=0,row=0,tria=0,trib=0,a2d=0,d2aA=0,d2aB=0,P=0,I=0,FBA=0,FBB=0,ARL=0,data_packet=0,dynamic=1):
        """
        setdfbrow(self,col=0,row=0,tria=0,trib=0,a2d=0,d2aA=0,d2aB=0,P=0,I=0,FBA=0,FBB=0,ARL=0,data_packet=0,dynamic=1)
        set every possibly setting for a single dfbx2 row, all values have defaults of 0 or off, or FBA,ERR.
        dynamic has default on so that everything gets sent.
        goes through the gui so everything stays in sync
        """
        dfbrap=self.dfbraps[col]
        if row == "master":
            chn = dfbrap.master_vector
        else:
            chn = dfbrap.state_vectors[row]
        chn.lock_button.setChecked(dynamic) #this is the dynamic button, when true all commands are send when gui is changed, the send all button should not be needed, do this first so I don't need to send manually
        chn.a2d_lockpt_spin.setValue(a2d)
        chn.d2a_A_spin.setValue(d2aA)
        chn.d2a_B_spin.setValue(d2aB)
        chn.P_spin.setValue(P)
        chn.I_spin.setValue(I)
        chn.data_packet.setCurrentIndex(data_packet)
        chn.TriA_button.setChecked(tria)
        chn.TriB_button.setChecked(trib)
        chn.ARL_button.setChecked(ARL)
        chn.FBA_button.setChecked(FBA)
        chn.FBB_button.setChecked(FBB)

    def setdfballrow(self,col=0,tria=0,trib=0,a2d=0,d2aA=0,d2aB=0,P=0,I=0,FBA=0,FBB=0,ARL=0,data_packet=0,dynamic=1):
        #rows = range(len(self.dfbraps[col].state_vectors))+["master"]
        rows = list(range(self.seqln))
        for row in rows:
            self.setdfbrow(col,row,tria,trib,a2d,d2aA,d2aB,P,I,FBA,FBB,ARL,data_packet,dynamic)

    def setdfball(self,tria=0,trib=0,a2d=0,d2aA=0,d2aB=0,P=0,I=0,FBA=0,FBB=0,ARL=0,data_packet=0,dynamic=1):
        for col in range(len(self.dfbraps)):
            self.setdfballrow(col,tria,trib,a2d,d2aA,d2aB,P,I,FBA,FBB,ARL,data_packet,dynamic)

    def setdfbrow_d2a(self, col, row, d2aA):
        dfbrap=self.dfbraps[col]
        if row == "master":
            chn = dfbrap.master_vector
        else:
            chn = dfbrap.state_vectors[row]
        chn.d2a_A_spin.setValue(d2aA)

    def settriangleparams(self,dwell=0,steps=10,stepsize=8,timebase=1):
        # timebase = 0 gives lsync, timebase = 1 gives frame
        # frame is wanted for all tuning,it makes all rows have same values in triangle
        self.cringe.dwell.setValue(dwell)
        self.cringe.range.setValue(steps) # carl has some annoyingly inconsistent names
        self.cringe.step.setValue(stepsize)
        self.cringe.tri_idx_button.setChecked(timebase)

    def settiming(self,sett=12,dfbpropdelay=3, dfbcarddelay=0, bad16carddelay=5):
        self.cringe.SETT_spin.setValue(sett)
        self.cringe.prop_delay_spin.setValue(dfbpropdelay)
        self.cringe.dfb_delay_spin.setValue(dfbcarddelay)
        self.cringe.bad_delay_spin.setValue(bad16carddelay)

    # carl made a set of timers so that some values are sent only after
    # the gui element has changed, but then not changed for a few seconds
    # here we call the change_blah functions to makesure we don't wait for the timer
    def setlsync(self, lsync):
        if lsync != self.lsync:
            self.cringe.lsync_spin.setValue(lsync)
            self.cringe.change_lsync()

    def setnsamp(self, nsamp):
        if nsamp != self.NSAMP:
            self.cringe.NSAMP_spin.setValue(nsamp)
            self.cringe.change_NSAMP()

    def setSETT(self, sett):
        if sett != self.SETT:
            self.cringe.SETT_spin.setValue(sett)
            self.cringe.change_SETT()

    def setFluxJumpThreshold(self,flux_jump_threshold):
        flux_jump_threshold= int(flux_jump_threshold)
        if flux_jump_threshold != self.flux_jump_threshold:
            self.cringe.ARLsense_spin.setValue(flux_jump_threshold)
            self.cringe.change_ARLsense()

    @property
    def seqln(self):
        return self.cringe.seqln
    @property
    def lsync(self):
        return self.cringe.lsync
    @property
    def SETT(self):
        return self.cringe.SETT
    @property
    def NSAMP(self):
        return self.cringe.NSAMP
    @property
    def prop_delay(self):
        return self.cringe.prop_delay
    @property
    def dfb_delay(self):
        return self.cringe.dfb_delay
    @property
    def bad_delay(self):
        return self.cringe.bad_delay

    @property
    def flux_jump_threshold(self):
        return self.cringe.ARLsense_spin.value()