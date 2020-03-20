'''
Created on July 20, 2011

@author: schimaf

Tower Power Supply controller class based on using two (2) Agilent E3631a GPIB power supplies
'''

#import agilent_e3631a

class TowerPowerSupplies(object):

    def __init__(self, pad1=5, pad2=6):

        '''
        Constructor
        '''
        try:
            import agilent_e3631a_serial
            self.power_supply_1 = agilent_e3631a_serial.AgilentE3631A("towerps1")
            self.power_supply_2 = agilent_e3631a_serial.AgilentE3631A("towerps2")
            print("using agilent_e3631a_serial, expect to wait 10 seconds for the GUI to work")
        except:
            print("failed to find working power supplies on serial ports towerps1 and towerps2")
            print("trying gpib")
            import agilent_e3631a
            self.pad1 = pad1
            self.pad2 = pad2
            self.power_supply_1 = agilent_e3631a.AgilentE3631A(pad=self.pad1)
            self.power_supply_2 = agilent_e3631a.AgilentE3631A(pad=self.pad2)




        self.powered = self.checkPowered()

        self.kdVa  =  6.00 # init settings needed by POWER card for +Va,-Va
        self.kdVr  =  9.00 # +Vr
        self.kdVd  =  6.00 # +Vd
        self.kdVoltageTolerance = 0.100 # for out-of-spec tests

        self.kdVa_limit_amps  = 0.300 # I_P25V_limit and I_N25V_limit
        self.kdVr_limit_amps  = 0.030 # I_P25V_limit
        self.kdVd_limit_amps  = 0.400 # limit on +Vd (I_P6V_limit)

    def checkPowered(self):

        v1 = self.power_supply_1.measureVoltage("P25V")
        result = False
        if v1 > 1:
            result = True

        return result

    def powerOnSequence(self):

        self.powerOnSupplies()
        self.setCurrentLimits()
        return self.measureVoltagesAndCurrents()

    def powerOnSupplies(self):

        self.power_supply_1.powerOn()
        self.power_supply_2.powerOn()
        self.powered = True

    def powerOffSupplies(self):

        self.power_supply_1.powerOff()
        self.power_supply_2.powerOff()
        self.powered = False

    def setCurrentLimits(self):

        self.power_supply_1.setCurrentLimit("P25V", self.kdVa, self.kdVa_limit_amps)
        self.power_supply_1.setCurrentLimit("N25V", -1 * self.kdVa, self.kdVa_limit_amps)

        self.power_supply_2.setCurrentLimit("P6V", self.kdVd, self.kdVd_limit_amps)
        self.power_supply_2.setCurrentLimit("P25V", self.kdVr, self.kdVr_limit_amps)

    def measureVoltagesAndCurrents(self):

        # Power Supply 2

        #Plus minus tolerance checks for +Vd Rail
        dV_msrd = self.power_supply_2.measureVoltage("P6V")
        if dV_msrd < ( self.kdVd - self.kdVoltageTolerance ) :
            sV_msrd = "%6.3f" %( dV_msrd )
            print "*** 'SetE3631': ERROR: Msrd +Vd (" + sV_msrd + ") TOO LOW; SHUTTING OUTPUTS OFF"
            self.powerOffSupplies()

        if dV_msrd > ( self.kdVd + self.kdVoltageTolerance ) :
            sV_msrd = "%6.3f" %( dV_msrd )
            print "*** 'SetE3631': ERROR: Msrd +Vd (" + sV_msrd + ") TOO HIGH; SHUTTING OUTPUTS OFF"
            self.powerOffSupplies()

        #Display Current for +Vd rail

        dI_msrd = self.power_supply_2.measureCurrent("P6V")
	s = "+Vd current = %6.3f_amps" % ( dI_msrd )
	print s


       #PLus minus tolerance checks for +Vr Rail

        dV_msrd = self.power_supply_2.measureVoltage("P25V")
        if dV_msrd < ( self.kdVr - self.kdVoltageTolerance ) :
            sV_msrd = "%6.3f" % ( dV_msrd )
            print "*** 'SetE3631': ERROR: Msrd +Vr (" + sV_msrd + ") TOO LOW; SHUTTING OUTPUTS OFF"
            self.powerOffSupplies()

        if dV_msrd > ( self.kdVr + self.kdVoltageTolerance ) :
            sV_msrd = "%6.3f" % ( dV_msrd )
            print "*** 'SetE3631': ERROR: Msrd +Vr (" + sV_msrd + ") TOO HIGH; SHUTTING OUTPUTS OFF"
            self.powerOffSupplies()

        #Display Current for +Vr Rail

        dI_msrd = self.power_supply_2.measureCurrent("P25V")
        s0 = "+Vr current = %6.3f_amps" % ( dI_msrd )
	print s0
	s+="\n"+s0
        #Power Supply 1

        #Plus minus tolerance checks for +Va Rail
        dV_msrd = self.power_supply_1.measureVoltage("P25V")
        if dV_msrd < ( self.kdVa - self.kdVoltageTolerance ) :
            print "meas +Va = %f +va = %f tol = %f" % (dV_msrd, self.kdVa, self.kdVoltageTolerance)
            sV_msrd = "%6.3f" %( dV_msrd )
            print "*** 'SetE3631': ERROR: Msrd +Va (" + sV_msrd + ") TOO LOW; SHUTTING OUTPUTS OFF"
            self.powerOffSupplies()

        #Display Current for +Va Rail
        dI_msrd = self.power_supply_1.measureCurrent("P25V")
        s0 = "+Va current = %6.3f_amps" % ( dI_msrd )
	print s0
	s+="\n"+s0

        #PLus minus tolerance checks for -Va rail
        dV_msrd = self.power_supply_1.measureVoltage("N25V")
        if dV_msrd > ( self.kdVoltageTolerance - self.kdVa ) :
            print "meas -Va = %f -va = %f tol = %f" % (dV_msrd, self.kdVa, self.kdVoltageTolerance)
            sV_msrd = "%6.3f" % ( dV_msrd )
            print "*** 'SetE3631': ERROR: Msrd -Va (" + sV_msrd + ") TOO LOW; SHUTTING OUTPUTS OFF"
            self.powerOffSupplies()

        if dV_msrd > (self.kdVa - self.kdVoltageTolerance ) :
            sV_msrd = "%6.3f" % ( dV_msrd )
            print "*** 'SetE3631': ERROR: Msrd -Va (" + sV_msrd + ") TOO HIGH; SHUTTING OUTPUTS OFF"
            self.powerOffSupplies()

        #Display Current for -Va rail

        dI_msrd = self.power_supply_1.measureCurrent("N25V")
        s0 = "-Va current = %6.3f_amps" % ( dI_msrd )
	print s0
	s+="\n"+s0
	return s
