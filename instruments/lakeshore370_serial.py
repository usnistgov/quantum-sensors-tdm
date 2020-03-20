'''
Lakesore370
Created on Mar 11, 2009
@author: bennett
'''

from lookup import Lookup
from time import sleep
import math
import numpy
#from scipy.io import read_array #obsolete, replace with numpy.genfromtxt
import pylab
import scipy
from scipy.interpolate import interp1d
from tkSimpleDialog import askfloat
import serial_instrument
import serial

class Lakeshore370(serial_instrument.SerialInstrument):
    '''
    The Lakeshore 370 AC Bridge GPIB communication class
    '''


    def __init__(self, port="lakeshore"):
        '''Constructor  The PAD (Primary GPIB Address) is the only required parameter '''

        super(Lakeshore370, self).__init__(port, bytesize=serial.SEVENBITS, parity=serial.PARITY_ODD,
        stopbits=serial.STOPBITS_ONE, min_time_between_writes=0.05, readtimeout=0.05)

        # GPIB identity string of the instrument
        self.id_string = "LSCI,MODEL370,370447,09272005"
        self.manufacturer = 'Lakeshore'
        self.model_number = '370'
        self.description  = 'Bridge - Temperature Controller'

        self.voltage = None

        #self.compare_identity()

        self.control_mode_switch = Lookup({
            'closed' : '1',
            'zone' : '2',
            'open' : '3',
            'off' : '4'
            })

        self.on_off_switch = Lookup({
            'off' : '0',
            'on' : '1'
            })

    def getTemperature(self, channel=1):
        ''' Get temperature from a given channel as a float '''

        commandstring = 'RDGK? ' + str(channel)
        #result = self.ask(commandstring)
        #self.voltage = float(result)
        self.voltage = self.askFloat(commandstring)

        return self.voltage

    def getResistance(self, channel=1):
        '''Get resistance from a given channel as a float.'''

        commandstring = 'RDGR? ' + str(channel)
#        result = self.ask(commandstring)
#        resistance = float(result)
        resistance = self.askFloat(commandstring)

        return resistance

    def setControlMode(self, controlmode = 'off'):
        ''' Set control mode 'off', 'zone', 'open' or 'closed' '''

        #switch = {
        #    'closed' : '1',
        #    'zone' : '2',
        #    'open' : '3',
        #    'off' : '4'
        #}

        #commandstring = 'CMODE ' + switch.get(controlmode,'4')
        commandstring = 'CMODE ' + self.control_mode_switch.get(controlmode,'4')
        self.write(commandstring)

    def getControlMode(self):
        ''' Get control mode 'off', 'zone', 'open' or 'closed' '''

        #switch = {
        #    '1' : 'closed',
        #    '2' : 'zone',
        #    '3' : 'open',
        #    '4' : 'off'
        #}

        commandstring = 'CMODE?'
        result = self.ask(commandstring).rstrip()
        #mode = switch.get(result, 'com error')
        mode = self.control_mode_switch.get_key(result)

        return mode[0]

    def setPIDValues(self, P=1, I=1, D=0):
        ''' Set P, I and D values where I and D are i nunits of seconds '''

        commandstring = 'PID ' + str(P) + ', ' + str(I) + ', ' + str(D)
        self.write(commandstring)

    def getPIDValues(self):
        '''Returns P,I and D values as floats where is I and D have units of seconds '''

        commandstring = 'PID?'
        result = self.ask(commandstring)
        valuestrings = result.split(',')
        PIDvalues = [0,0,0]
        PIDvalues[0] = float(valuestrings[0])
        PIDvalues[1] = float(valuestrings[1])
        PIDvalues[2] = float(valuestrings[2])

        return PIDvalues

    def setManualHeaterOut(self, heatpercent=0):
        ''' Set the manual heater output as a percent of heater range '''

        commandstring = 'MOUT ' + str(heatpercent)
        self.write(commandstring)

    def getManualHeaterOut(self):
        ''' Get the manual heater output as a percent of heater range '''

        commandstring = 'MOUT?'

        result = self.ask(commandstring)
        heaterout = float(result)

        return heaterout

    def getHeaterOut(self):
        ''' Get the manual heater output as a percent of heater range '''

        commandstring = 'HTR?'

        result = self.ask(commandstring)
        heaterout = float(result)

        return heaterout

    def setTemperatureSetPoint(self, setpoint=0.010):
        ''' Set the temperature set point in units of Kelvin '''

        commandstring = 'SETP ' + str(setpoint)
        self.write(commandstring)

    def getTemperatureSetPoint(self):
        ''' Get the temperature set point in units of Kelvin '''

        commandstring = 'SETP?'
        result = self.ask(commandstring)
        setpoint = float(result)

        return setpoint

    def setHeaterRange(self, range=10):
        ''' Set the temperature heater range in units of mA '''

        if range >= 0.0316 and range < .1:
            rangestring = '1'
        elif range >= .1 and range < .316:
            rangestring = '2'
        elif range >= .316 and range < 1:
            rangestring = '3'
        elif range >= 1 and range < 3.16:
            rangestring = '4'
        elif range >= 3.16 and range < 10:
            rangestring = '5'
        elif range >= 10 and range < 31.6:
            rangestring = '6'
        elif range >= 31.6 and range < 100:
            rangestring = '7'
        elif range >= 100 and range < 316:
            rangestring = '8'
        else:
            rangestring = '0'

        commandstring = 'HTRRNG ' + str(rangestring)
        result = self.write(commandstring)

    def getHeaterRange(self):
        ''' Get the temperature heater range in units of mA '''

        switch = {
            '0' : 0,
            '1' : 0,
            '2' : 0.100,
            '3' : 0.316,
            '4' : 1,
            '5' : 3.16,
            '6' : 10,
            '7' : 31.6,
            '8' : 100
            }

        commandstring = 'HTRRNG?'
        result = self.ask(commandstring)
        htrrange = switch.get(result , 'com error')

        return htrrange

    def setControlPolarity(self, polarity = 'unipolar'):
        ''' Set the heater output polarity 'unipolar' or 'bipolar' '''

        switch = {
            'unipolar' : '0',
            'bipolar' : '1'
        }

        commandstring = 'CPOL ' + switch.get(polarity,'0')
        self.write(commandstring)

    def getControlPolarity(self):
        ''' Get the heater output polarity 'unipolar' or 'bipolar' '''

        switch = {
            '0' : 'unipolar',
            '1' : 'bipolar'
        }

        commandstring = 'CPOL?'
        result = self.ask(commandstring)
        polarity = switch.get(result , 'com error')

        return polarity

    def setScan(self, channel = 1, autoscan = 'off'):
        ''' Set the channel autoscanner 'on' or 'off' '''

        switch = {
            'off' : '0',
            'on' : '1'
        }

        commandstring = 'SCAN ' + str(channel) + ', ' + switch.get(autoscan,'0')
        self.write(commandstring)

    def setRamp(self, rampmode = 'on' , ramprate = 0.1):
        ''' Set the ramp mode to 'on' or 'off' and specify ramp rate in Kelvin/minute'''

        switch = {
            'off' : '0',
            'on' : '1'
        }

        commandstring = 'RAMP ' + switch.get(rampmode,'1') + ', ' + str(ramprate)
        self.write(commandstring)


    def getRamp(self):
        ''' Get the ramp mode either 'on' or 'off' and the ramp rate in Kelvin/minute '''

        switch = {
            '0' : 'off',
            '1' : 'on'
        }

        commandstring = 'RAMP?'
        result = self.ask(commandstring)
        results = result.split(',')
        ramp = ['off', 0]
        ramp[0] = switch.get(results[0] , 'com error')
        ramp[1] = float(results[1])

        return ramp

    def setTemperatureControlSetup(self, channel = 1, units = 'Kelvin', maxrange = 10, delay = 2, htrres = 1, output = 'current', filterread = 'unfiltered'):
        '''
        Setup the temperature control channel, units 'Kelvin' or 'Ohms', the maximum heater range in mA, delay in seconds, heater resistance in Ohms, output the 'current' or 'power', and 'filterer' or 'unfiltered'
        '''

        switchunits = {
            'Kelvin' : '1',
             'Ohms' : '2'
        }

        if maxrange >= 0.0316 and maxrange < .1:
            rangestring = '1'
        elif maxrange >= .1 and maxrange < .316:
            rangestring = '2'
        elif maxrange >= .316 and maxrange < 1:
            rangestring = '3'
        elif maxrange >= 1 and maxrange < 3.16:
            rangestring = '4'
        elif maxrange >= 3.16 and maxrange < 10:
            rangestring = '5'
        elif maxrange >= 10 and maxrange < 31.6:
            rangestring = '6'
        elif maxrange >= 31.6 and maxrange < 100:
            rangestring = '7'
        elif maxrange >= 100 and maxrange < 316:
            rangestring = '8'
        else:
            rangestring = '0'

        switchoutput = {
            'current' : '1',
            'power' : '2'
        }

        switchfilter = {
            'unfiltered' : '0',
            'filtered' : '1'
        }

        commandstring = 'CSET ' + str(channel) + ', ' + switchfilter.get(filterread,'0') + ', ' + switchunits.get(units,'1') + ', ' + str(delay) + ', ' + switchoutput.get(output,'1') + ', ' + rangestring + ', ' + str(htrres)
        self.write(commandstring)

    def setReadChannelSetup(self, channel = 1, mode = 'current', exciterange = 10e-9, resistancerange = 63.2e3,autorange = 'off', excitation = 'on'):
        '''
        Sets the measurment parameters for a given channel, in 'current' or 'voltage' excitation mode, excitation range in Amps or Volts, resistance range in ohms
        '''

        switchmode = {
            'voltage' : '0',
            'current' : '1'
        }

        switchautorange = {
            'off' : '0',
            'on' : '1'
        }

        switchexcitation = {
            'on' : '0',
            'off' : '1'
        }


        #Get Excitation Range String
        if mode == 'voltage':
            if exciterange >= 2e-6 and exciterange < 6.32e-6:
                exciterangestring = '1'
            elif exciterange >= 6.32e-6 and exciterange < 20e-6:
                exciterangestring = '2'
            elif exciterange >= 20e-6 and exciterange < 63.2e-6:
                exciterangestring = '3'
            elif exciterange >= 63.2e-6 and exciterange < 200e-6:
                exciterangestring = '4'
            elif exciterange >= 200e-6 and exciterange < 632e-6:
                exciterangestring = '5'
            elif exciterange >= 632e-6 and exciterange < 2e-3:
                exciterangestring = '6'
            elif exciterange >= 2e-3 and exciterange < 6.32e-3:
                exciterangestring = '7'
            elif exciterange >= 6.32e-3 and exciterange < 20e-3:
                exciterangestring = '8'
            else:
                exciterangestring = '1'
        else:
            if exciterange >= 1e-12 and exciterange < 3.16e-12:
                exciterangestring = '1'
            elif exciterange >= 3.16e-12 and exciterange < 10e-12:
                exciterangestring = '2'
            elif exciterange >= 10e-12 and exciterange < 31.6e-12:
                exciterangestring = '3'
            elif exciterange >= 31.6e-12 and exciterange < 100e-12:
                exciterangestring = '4'
            elif exciterange >= 100e-12 and exciterange < 316e-12:
                exciterangestring = '5'
            elif exciterange >= 316e-12 and exciterange < 1e-9:
                exciterangestring = '6'
            elif exciterange >= 1e-9 and exciterange < 3.16e-9:
                exciterangestring = '7'
            elif exciterange >= 3.16e-9 and exciterange < 10e-9:
                exciterangestring = '8'
            elif exciterange >= 10e-9 and exciterange < 31.6e-9:
                exciterangestring = '9'
            elif exciterange >= 31.6e-9 and exciterange < 100e-9:
                exciterangestring = '10'
            elif exciterange >= 100e-9 and exciterange < 316e-9:
                exciterangestring = '11'
            elif exciterange >= 316e-9 and exciterange < 1e-6:
                exciterangestring = '12'
            elif exciterange >= 1e-6 and exciterange < 3.16e-6:
                exciterangestring = '13'
            elif exciterange >= 3.16e-6 and exciterange < 10e-6:
                exciterangestring = '14'
            elif exciterange >= 10e-6 and exciterange < 31.6e-6:
                exciterangestring = '15'
            elif exciterange >= 31.6e-6 and exciterange < 100e-6:
                exciterangestring = '16'
            elif exciterange >= 100e-6 and exciterange < 316e-6:
                exciterangestring = '17'
            elif exciterange >= 316e-6 and exciterange < 1e-3:
                exciterangestring = '18'
            elif exciterange >= 1e-3 and exciterange < 3.16e-3:
                exciterangestring = '19'
            elif exciterange >= 3.16e-3 and exciterange < 10e-3:
                exciterangestring = '20'
            elif exciterange >= 10e-3 and exciterange < 31.6e-3:
                exciterangestring = '21'
            elif exciterange >= 31.6e-3 and exciterange < 100e-3:
                exciterangestring = '22'
            else:
                exciterangestring = '7'

            #Get Resistance Range String
        if resistancerange < 2e-3:
            resistancerangestring= '1'
        elif resistancerange > 2e-3 and resistancerange <= 6.32e-3:
            resistancerangestring = '2'
        elif resistancerange > 6.32e-3 and resistancerange <= 20e-3:
            resistancerangestring = '3'
        elif resistancerange > 20e-3 and resistancerange <= 63.2e-3:
            resistancerangestring = '4'
        elif resistancerange > 63.2e-3 and resistancerange <= 200e-3:
            resistancerangestring = '5'
        elif resistancerange > 200e-3 and resistancerange <= 632e-3:
            resistancerangestring = '6'
        elif resistancerange > 632e-3 and resistancerange <= 2.0:
            resistancerangestring = '7'
        elif resistancerange > 2.0 and resistancerange <= 6.32:
            resistancerangestring = '8'
        elif resistancerange > 6.32 and resistancerange <= 20:
            resistancerangestring = '9'
        elif resistancerange > 20 and resistancerange <= 63.2:
            resistancerangestring = '10'
        elif resistancerange > 63.2 and resistancerange <= 200:
            resistancerangestring = '11'
        elif resistancerange > 200 and resistancerange <= 632:
            resistancerangestring = '12'
        elif resistancerange > 632 and resistancerange <= 2e3:
            resistancerangestring = '13'
        elif resistancerange > 2e3 and resistancerange <= 6.32e3:
            resistancerangestring = '14'
        elif resistancerange > 6.32e3 and resistancerange <= 20e3:
            resistancerangestring = '15'
        elif resistancerange > 20e3 and resistancerange <= 63.2e3:
            resistancerangestring = '16'
        elif resistancerange > 63.2e3 and resistancerange <= 200e3:
            resistancerangestring = '17'
        elif resistancerange > 200e3 and resistancerange <= 632e3:
            resistancerangestring = '18'
        elif resistancerange > 632e3 and resistancerange <= 2e6:
            resistancerangestring = '19'
        elif resistancerange > 2e6 and resistancerange <= 6.32e6:
            resistancerangestring = '20'
        elif resistancerange > 6.32e6 and resistancerange <= 20e6:
            resistancerangestring = '21'
        elif resistancerange > 20e6 and resistancerange <= 63.2e6:
            resistancerangestring = '22'
        elif resistancerange > 63.2e6 and resistancerange <= 200e6:
            resistancerangestring = '23'
        else:
            resistancerangestring = '1'

        #Send Resistance Range Command String
        commandstring = 'RDGRNG ' + str(channel) + ', ' + switchmode.get(mode,'1') + ', ' + exciterangestring + ',' + resistancerangestring + ',' + switchautorange.get(autorange,'0') + ', ' + switchexcitation.get(excitation,'0')
        self.write(commandstring)

    def getHeaterStatus(self):

        switch = {
            '0' : 'no error',
            '1' : 'heater open error'
        }

        commandstring = 'HTRST?'
        result = self.ask(commandstring)
        status = switch.get(result, 'com error')

        return status

    def magUpSetup(self, heater_resistance=1):
        ''' Setup the lakeshore for magup '''

        self.setTemperatureControlSetup(channel=1, units='Kelvin', maxrange=10, delay=2, htrres=heater_resistance, output='current', filterread='unfiltered')
        self.setControlMode(controlmode = 'open')
        self.setControlPolarity(polarity = 'unipolar')
        self.setHeaterRange(range=10) 	# 1 Volt max input to Kepco for 100 Ohm shunt
        self.setReadChannelSetup(channel = 1, mode = 'current', exciterange = 10e-9, resistancerange = 2e3,autorange = 'on')


    def demagSetup(self, heater_resistance=1):
        ''' Setup the lakeshore for demag '''

        self.setTemperatureControlSetup(channel=1, units='Kelvin', maxrange=10, delay=2, htrres=heater_resistance, output='current', filterread='unfiltered')
        self.setControlMode(controlmode = 'open')
        self.setControlPolarity(polarity = 'bipolar')  #Set to bipolar so that current can get to zero faster
        self.setHeaterRange(range=10)  # 1 Volt max input to Kepco for 100 Ohm shunt
        self.setReadChannelSetup(channel = 1, mode = 'current', exciterange = 10e-9, resistancerange = 2e3,autorange = 'on')


    def setupPID(self, exciterange=3.16e-9, therm_control_channel=1, ramprate=0.05, heater_resistance=1):
        '''Setup the lakeshore for temperature regulation '''
        self.setScan(channel = therm_control_channel, autoscan = 'off')
        sleep(3)
        self.setReadChannelSetup(channel=1, mode='current', exciterange=exciterange, resistancerange=63.2e3,autorange='on')
        sleep(15)  #Give time for range to settle, or servoing will fail
        self.setReadChannelSetup(channel=1, mode='current', exciterange=exciterange, resistancerange=63.2e3,autorange='off')
        sleep(2)
        self.setTemperatureControlSetup(channel=1, units='Kelvin', maxrange=100, delay=2, htrres=heater_resistance, output='current', filterread='unfiltered')
        self.setControlMode(controlmode = 'closed')
        self.setControlPolarity(polarity = 'unipolar')
        self.setRamp(rampmode = 'off') #Turn off ramp mode to not to ramp setpoint down to aprox 0
        sleep(.5) #Give time for Set Ramp to take effect
        self.SetTemperatureSetPoint(setpoint=0.035)
        sleep(.5) #Give time for Setpoint to take effect
        self.setRamp(rampmode = 'on' , ramprate = ramprate)
        self.setHeaterRange(range=100) #Set heater range to 100mA to get 10V output range
        #self.SetReadChannelSetup(channel = 1, mode = 'current', exciterange = 1e-9, resistancerange = 2e3,autorange = 'on')

# Public Calibration Methods

    def sendStandardRuOxCalibration(self):
        pass

    def sendCalibrationFromArrays(self, rData, tData, curveindex, thermname='Cernox 1030', serialnumber='x0000',\
                            temp_lim=300, tempco = 1, units=4, makeFig = False):
        ''' Send a calibration based on a input file

            Input:
            rData: array of themometer resistance values (Ohms for units=3 or log(R/Ohms) for units=4)
            tData: array of themometer temperature values (Kelvin)
            curveindex: the curve index location in the lakeshore 370 (1 thru 20)
            thermname: sensor type
            serialnumber: thermometer serial number
            interp: if True the data will be evenly spaced from the max to min with 200 pts
                    if False the raw data is used.  User must ensure no doubles and < 200 pts
            temp_lim: temperature limit (K)
            tempco: 1 if dR/dT is negative, 2 if dR/dT is positive
            units: 3 to use ohm/K, 4 to use log ohm/K
        '''

        if curveindex < 1 or curveindex > 20:
            print ' 1 <= curveindex <= 20 for lakeshore 370'
            return 1




        # Send Header
        # 4, 350 ,1 -- logOhm/K, temperature limit, temperature coefficient 1=negative
        commandstring = 'CRVHDR ' + str(curveindex) + ', ' + thermname + ', ' + serialnumber + ', '+str(units)+', '+\
            str(temp_lim)+', '+ str(tempco)
        self.write(commandstring)
        print(commandstring)

        # Send Data Points
        for i in range(len(rData)):
            pntindex = i+1

            if rData[i] < 10:
                stringRPoint = '%7.5f' % rData[i]
            else:
                stringRPoint = '%8.5f' % rData[i]

            stringTPoint = '%7.5f' % tData[i]

            datapointstring = 'CRVPT ' + str(curveindex) + ', ' + str(pntindex) + ', ' + stringRPoint + ', ' + stringTPoint
            self.write(datapointstring)
            print datapointstring

        if makeFig:
            pylab.figure()
            pylab.plot(rData,tData,'o')
            pylab.xlabel('Resistance (Ohms)')
            pylab.ylabel('Temperature (K)')

    def sendCalibration(self, filename, datacol, tempcol, curveindex, thermname='Cernox 1030', serialnumber='x0000', interp=True,\
                            temp_lim=300, tempco = 1, units=4):
        ''' Send a calibration based on a input file

            Input:
            filename: location of calibration file
            datacol: defines which column in filename will be used as data (zero indexed)
            tempcol: defines which column in filename will be used as temperature (zero indexed)
            curveindex: the curve index location in the lakeshore 370 (1 thru 20)
            thermname: sensor type
            serialnumber: thermometer serial number
            interp: if True the data will be evenly spaced from the max to min with 200 pts
                    if False the raw data is used.  User must ensure no doubles and < 200 pts
            temp_lim: temperature limit (K)
            tempco: 1 if dR/dT is negative, 2 if dR/dT is positive
            units: 3 to use ohm/K, 4 to use log ohm/K
        '''

        if curveindex < 1 or curveindex > 20:
            print ' 1 <= curveindex <= 20 for lakeshore 370'
            return 1

        #rawdata = read_array(filename) #obsolete, replace with genfromtxt
	rawdata = numpy.genfromtxt(filename)
        rawdatat = rawdata.transpose()
        datat = numpy.array(rawdatat[:,rawdatat[datacol,:].argsort()])

        #now remove doubles
        last = datat[datacol,-1]

        for i in range(len(datat[datacol,:])-2,-1,-1):
            if last == datat[1,i]:
                datat = numpy.hstack((datat[:,: i+1],datat[:,i+2 :]))
            else:
                last = datat[datacol,i]

        pylab.figure()
        pylab.plot(datat[datacol],datat[tempcol],'o')
        pylab.show()

        f = interp1d(datat[datacol],datat[tempcol])
        self.f = f

        # interpolate from min to max with 200 evenly spaced points if interp True
        if interp:
            Rs = scipy.linspace(min(datat[datacol]),max(datat[datacol]), num = 200)
        else:
            Rs = datat[datacol]
	Rs[1] = 2730
	Rs[2] = 2930
	Rs[3] = 3100
        Temps = f(Rs)

        pylab.figure()
        pylab.plot(datat[datacol],datat[tempcol],'o')
        #pylab.holdon()
        pylab.plot(Rs,Temps,'rx')
        pylab.show()

        # Send Header
        # 4, 350 ,1 -- logOhm/K, temperature limit, temperature coefficient 1=negative
        commandstring = 'CRVHDR ' + str(curveindex) + ', ' + thermname + ', ' + serialnumber + ', '+str(units)+', '+\
            str(temp_lim)+', '+ str(tempco)
        self.write(commandstring)
        print commandstring

        # Send Data Points
        for i in range(len(Rs)):
            pntindex = i+1
            if units == 4:
                logrofpoint = math.log10(Rs[i])
            else:
                logrofpoint = Rs[i]

            if Rs[i] < 10:
                stringlogrofpoint = '%(logrofpoint)7.5f' % vars()
            else:
                stringlogrofpoint = '%(logrofpoint)8.5f' % vars()

            tempofpoint = Temps[i]
            stringtempofpoint = '%(tempofpoint)5.5f' % vars()

            datapointstring = 'CRVPT ' + str(curveindex) + ', ' + str(pntindex) + ', ' + stringlogrofpoint + ', ' + stringtempofpoint
            self.write(datapointstring)
            print datapointstring

        pylab.figure()
        pylab.plot(Rs,Temps,'o')
        pylab.xlabel('Resistance (Ohms)')
        pylab.ylabel('Temperature (K)')

    def sendMartinisRuOxCalibration(self, curveindex, thermname='RuOx Martinis', serialnumber='19740', interp=True,\
                            temp_lim=300, tempco = 1, units=4):
        self.sendMartinisCalibration(curveindex, thermname, serialnumber, interp, temp_lim, tempco, units)

    def sendMartinisCalibration(self, curveindex, thermname='RuOx Martinis', serialnumber='19740', interp=True,\
                            temp_lim=300, tempco = 1, units=4):
        ''' Send a calibration based on a input file

            Input:
            filename: location of calibration file
            datacol: defines which column in filename will be used as data (zero indexed)
            tempcol: defines which column in filename will be used as temperature (zero indexed)
            curveindex: the curve index location in the lakeshore 370 (1 thru 20)
            thermname: sensor type
            serialnumber: thermometer serial number
            interp: if True the data will be evenly spaced from the max to min with 200 pts
                    if False the raw data is used.  User must ensure no doubles and < 200 pts
            temp_lim: temperature limit (K)
            tempco: 1 if dR/dT is negative, 2 if dR/dT is positive
            units: 3 to use ohm/K, 4 to use log ohm/K
        '''

        if curveindex < 1 or curveindex > 20:
            print ' 1 <= curveindex <= 20 for lakeshore 370'
            return 1

        #rawdata = read_array(filename)
        #rawdatat = rawdata.transpose()
        #datat = numpy.array(rawdatat[:,rawdatat[datacol,:].argsort()])

        #now remove doubles
        #last = datat[datacol,-1]

        #for i in range(len(datat[datacol,:])-2,-1,-1):
        #    if last == datat[1,i]:
        #        datat = numpy.hstack((datat[:,: i+1],datat[:,i+2 :]))
        #    else:
        #        last = datat[datacol,i]

        #pylab.figure()
        #pylab.plot(datat[datacol],datat[tempcol],'o')
        #pylab.show()

        #f = interp1d(datat[datacol],datat[tempcol])
        #self.f = f

        # interpolate from min to max with 200 evenly spaced points if interp True
        #if interp:
        #Rs = scipy.linspace(min(datat[datacol]),max(datat[datacol]), num = 200)
        Rs = scipy.linspace(1000.92541, 63095.734448, num=200)
        #else:
        #    Rs = datat[datacol]
        Temps = (2.85 / (numpy.log((Rs-652.)/100.)))**4

#        pylab.figure()
#        #pylab.plot(datat[datacol],datat[tempcol],'o')
#        #pylab.holdon()
#        pylab.plot(Rs,Temps,'rx')
#        pylab.show()

        # Send Header
        # 4, 350 ,1 -- logOhm/K, temperature limit, temperature coefficient 1=negative
        commandstring = 'CRVHDR ' + str(curveindex) + ', ' + thermname + ', ' + serialnumber + ', '+str(units)+', '+\
            str(temp_lim)+', '+ str(tempco)
        self.write(commandstring)
        print commandstring

        # Send Data Points
        for i in range(len(Rs)):
            pntindex = i+1
            logrofpoint = math.log10(Rs[i])

            if Rs[i] < 10:
                stringlogrofpoint = '%(logrofpoint)7.5f' % vars()
            else:
                stringlogrofpoint = '%(logrofpoint)8.5f' % vars()

            tempofpoint = Temps[i]
            stringtempofpoint = '%(tempofpoint)7.5f' % vars()

            datapointstring = 'CRVPT ' + str(curveindex) + ', ' + str(pntindex) + ', ' + stringlogrofpoint + ', ' + stringtempofpoint
            self.write(datapointstring)
            print datapointstring

        pylab.figure()
        pylab.plot(Rs,Temps,'o')
        pylab.xlabel('Resistance (Ohms)')
        pylab.ylabel('Temperature (K)')
        pylab.show()



    # All these methods have been renamed and will be depricated eventualy

    def GetTemperature(self, channel=1):
        ''' Get temperature from a given channel as a float '''

        commandstring = 'RDGK? ' + str(channel)
        #result = self.ask(commandstring)
        #self.voltage = float(result)
        self.voltage = self.askFloat(commandstring)

        return self.voltage

    def GetResistance(self, channel=1):
        '''Get resistance from a given channel as a float.'''

        commandstring = 'RDGR? ' + str(channel)
#        result = self.ask(commandstring)
#        resistance = float(result)
        resistance = self.askFloat(commandstring)

        return resistance

    def SetControlMode(self, controlmode = 'off'):
        ''' Set control mode 'off', 'zone', 'open' or 'closed' '''

        #switch = {
        #    'closed' : '1',
        #    'zone' : '2',
        #    'open' : '3',
        #    'off' : '4'
        #}

        #commandstring = 'CMODE ' + switch.get(controlmode,'4')
        commandstring = 'CMODE ' + self.control_mode_switch.get(controlmode,'4')
        self.write(commandstring)

    def GetControlMode(self):
        ''' Get control mode 'off', 'zone', 'open' or 'closed' '''

        #switch = {
        #    '1' : 'closed',
        #    '2' : 'zone',
        #    '3' : 'open',
        #    '4' : 'off'
        #}

        commandstring = 'CMODE?'
        result = self.ask(commandstring)
        #mode = switch.get(result, 'com error')
        mode = self.control_mode_switch.get_key(result)

        return mode[0]

    def SetPIDValues(self, P=1, I=1, D=0):
        ''' Set P, I and D values where I and D are i nunits of seconds '''

        commandstring = 'PID ' + str(P) + ', ' + str(I) + ', ' + str(D)
        self.write(commandstring)

    def GetPIDValues(self):
        '''Returns P,I and D values as floats where is I and D have units of seconds '''

        commandstring = 'PID?'
        result = self.ask(commandstring)
        valuestrings = result.split(',')
        PIDvalues = [0,0,0]
        PIDvalues[0] = float(valuestrings[0])
        PIDvalues[1] = float(valuestrings[1])
        PIDvalues[2] = float(valuestrings[2])

        return PIDvalues

    def SetManualHeaterOut(self, heatpercent=0):
        ''' Set the manual heater output as a percent of heater range '''

        commandstring = 'MOUT ' + str(heatpercent)
        self.write(commandstring)

    def GetManualHeaterOut(self):
        ''' Get the manual heater output as a percent of heater range '''

        commandstring = 'MOUT?'

        result = self.ask(commandstring)
        heaterout = float(result)

        return heaterout

    def GetHeaterOut(self):
        ''' Get the manual heater output as a percent of heater range '''

        commandstring = 'HTR?'

        result = self.ask(commandstring)
        heaterout = float(result)

        return heaterout

    def SetTemperatureSetPoint(self, setpoint=0.010):
        ''' Set the temperature set point in units of Kelvin '''

        commandstring = 'SETP ' + str(setpoint)
        self.write(commandstring)

    def GetTemperatureSetPoint(self):
        ''' Get the temperature set point in units of Kelvin '''

        commandstring = 'SETP?'
        result = self.ask(commandstring)
        setpoint = float(result)

        return setpoint

    def SetHeaterRange(self, range=10):
        ''' Set the temperature heater range in units of mA '''

        if range >= 0.0316 and range < .1:
            rangestring = '1'
        elif range >= .1 and range < .316:
            rangestring = '2'
        elif range >= .316 and range < 1:
            rangestring = '3'
        elif range >= 1 and range < 3.16:
            rangestring = '4'
        elif range >= 3.16 and range < 10:
            rangestring = '5'
        elif range >= 10 and range < 31.6:
            rangestring = '6'
        elif range >= 31.6 and range < 100:
            rangestring = '7'
        elif range >= 100 and range < 316:
            rangestring = '8'
        else:
            rangestring = '0'

        commandstring = 'HTRRNG ' + str(rangestring)
        result = self.write(commandstring)

    def GetHeaterRange(self):
        ''' Get the temperature heater range in units of mA '''

        switch = {
            '0' : 0,
            '1' : 0,
            '2' : 0.100,
            '3' : 0.316,
            '4' : 1,
            '5' : 3.16,
            '6' : 10,
            '7' : 31.6,
            '8' : 100
            }

        commandstring = 'HTRRNG?'
        result = self.ask(commandstring)
        htrrange = switch.get(result , 'com error')

        return htrrange

    def SetControlPolarity(self, polarity = 'unipolar'):
        ''' Set the heater output polarity 'unipolar' or 'bipolar' '''

        switch = {
            'unipolar' : '0',
            'bipolar' : '1'
        }

        commandstring = 'CPOL ' + switch.get(polarity,'0')
        self.write(commandstring)

    def GetControlPolarity(self):
        ''' Get the heater output polarity 'unipolar' or 'bipolar' '''

        switch = {
            '0' : 'unipolar',
            '1' : 'bipolar'
        }

        commandstring = 'CPOL?'
        result = self.ask(commandstring)
        polarity = switch.get(result , 'com error')

        return polarity

    def SetScan(self, channel = 1, autoscan = 'off'):
        ''' Set the channel autoscanner 'on' or 'off' '''

        switch = {
            'off' : '0',
            'on' : '1'
        }

        commandstring = 'SCAN ' + str(channel) + ', ' + switch.get(autoscan,'0')
        self.write(commandstring)

    def SetRamp(self, rampmode = 'on' , ramprate = 0.1):
        ''' Set the ramp mode to 'on' or 'off' and specify ramp rate in Kelvin/minute'''

        switch = {
            'off' : '0',
            'on' : '1'
        }

        commandstring = 'RAMP ' + switch.get(rampmode,'1') + ', ' + str(ramprate)
        self.write(commandstring)


    def GetRamp(self):
        ''' Get the ramp mode either 'on' or 'off' and the ramp rate in Kelvin/minute '''

        switch = {
            '0' : 'off',
            '1' : 'on'
        }

        commandstring = 'RAMP?'
        result = self.ask(commandstring)
        results = result.split(',')
        ramp = ['off', 0]
        ramp[0] = switch.get(results[0] , 'com error')
        ramp[1] = float(results[1])

        return ramp

    def SetTemperatureControlSetup(self, channel = 1, units = 'Kelvin', maxrange = 10, delay = 2, htrres = 1, output = 'current', filterread = 'unfiltered'):
        '''
        Setup the temperature control channel, units 'Kelvin' or 'Ohms', the maximum heater range in mA, delay in seconds, heater resistance in Ohms, output the 'current' or 'power', and 'filterer' or 'unfiltered'
        '''

        switchunits = {
            'Kelvin' : '1',
             'Ohms' : '2'
        }

        if maxrange >= 0.0316 and maxrange < .1:
            rangestring = '1'
        elif maxrange >= .1 and maxrange < .316:
            rangestring = '2'
        elif maxrange >= .316 and maxrange < 1:
            rangestring = '3'
        elif maxrange >= 1 and maxrange < 3.16:
            rangestring = '4'
        elif maxrange >= 3.16 and maxrange < 10:
            rangestring = '5'
        elif maxrange >= 10 and maxrange < 31.6:
            rangestring = '6'
        elif maxrange >= 31.6 and maxrange < 100:
            rangestring = '7'
        elif maxrange >= 100 and maxrange < 316:
            rangestring = '8'
        else:
            rangestring = '0'

        switchoutput = {
            'current' : '1',
            'power' : '2'
        }

        switchfilter = {
            'unfiltered' : '0',
            'filtered' : '1'
        }

        commandstring = 'CSET ' + str(channel) + ', ' + switchfilter.get(filterread,'0') + ', ' + switchunits.get(units,'1') + ', ' + str(delay) + ', ' + switchoutput.get(output,'1') + ', ' + rangestring + ', ' + str(htrres)
        self.write(commandstring)

    def SetReadChannelSetup(self, channel = 1, mode = 'current', exciterange = 10e-9, resistancerange = 63.2e3,autorange = 'off', excitation = 'on'):
        '''
        Sets the measurment parameters for a given channel, in 'current' or 'voltage' excitation mode, excitation range in Amps or Volts, resistance range in ohms
        '''

        switchmode = {
            'voltage' : '0',
            'current' : '1'
        }

        switchautorange = {
            'off' : '0',
            'on' : '1'
        }

        switchexcitation = {
            'on' : '0',
            'off' : '1'
        }


        #Get Excitation Range String
        if mode == 'voltage':
            if exciterange >= 2e-6 and exciterange < 6.32e-6:
                exciterangestring = '1'
            elif exciterange >= 6.32e-6 and exciterange < 20e-6:
                exciterangestring = '2'
            elif exciterange >= 20e-6 and exciterange < 63.2e-6:
                exciterangestring = '3'
            elif exciterange >= 63.2e-6 and exciterange < 200e-6:
                exciterangestring = '4'
            elif exciterange >= 200e-6 and exciterange < 632e-6:
                exciterangestring = '5'
            elif exciterange >= 632e-6 and exciterange < 2e-3:
                exciterangestring = '6'
            elif exciterange >= 2e-3 and exciterange < 6.32e-3:
                exciterangestring = '7'
            elif exciterange >= 6.32e-3 and exciterange < 20e-3:
                exciterangestring = '8'
            else:
                exciterangestring = '1'
        else:
            if exciterange >= 1e-12 and exciterange < 3.16e-12:
                exciterangestring = '1'
            elif exciterange >= 3.16e-12 and exciterange < 10e-12:
                exciterangestring = '2'
            elif exciterange >= 10e-12 and exciterange < 31.6e-12:
                exciterangestring = '3'
            elif exciterange >= 31.6e-12 and exciterange < 100e-12:
                exciterangestring = '4'
            elif exciterange >= 100e-12 and exciterange < 316e-12:
                exciterangestring = '5'
            elif exciterange >= 316e-12 and exciterange < 1e-9:
                exciterangestring = '6'
            elif exciterange >= 1e-9 and exciterange < 3.16e-9:
                exciterangestring = '7'
            elif exciterange >= 3.16e-9 and exciterange < 10e-9:
                exciterangestring = '8'
            elif exciterange >= 10e-9 and exciterange < 31.6e-9:
                exciterangestring = '9'
            elif exciterange >= 31.6e-9 and exciterange < 100e-9:
                exciterangestring = '10'
            elif exciterange >= 100e-9 and exciterange < 316e-9:
                exciterangestring = '11'
            elif exciterange >= 316e-9 and exciterange < 1e-6:
                exciterangestring = '12'
            elif exciterange >= 1e-6 and exciterange < 3.16e-6:
                exciterangestring = '13'
            elif exciterange >= 3.16e-6 and exciterange < 10e-6:
                exciterangestring = '14'
            elif exciterange >= 10e-6 and exciterange < 31.6e-6:
                exciterangestring = '15'
            elif exciterange >= 31.6e-6 and exciterange < 100e-6:
                exciterangestring = '16'
            elif exciterange >= 100e-6 and exciterange < 316e-6:
                exciterangestring = '17'
            elif exciterange >= 316e-6 and exciterange < 1e-3:
                exciterangestring = '18'
            elif exciterange >= 1e-3 and exciterange < 3.16e-3:
                exciterangestring = '19'
            elif exciterange >= 3.16e-3 and exciterange < 10e-3:
                exciterangestring = '20'
            elif exciterange >= 10e-3 and exciterange < 31.6e-3:
                exciterangestring = '21'
            elif exciterange >= 31.6e-3 and exciterange < 100e-3:
                exciterangestring = '22'
            else:
                exciterangestring = '7'

            #Get Resistance Range String
        if resistancerange >= 2e-3 and resistancerange < 6.32e-3:
            resistancerangestring = '1'
        elif resistancerange >= 6.32e-3 and resistancerange < 20e-3:
            resistancerangestring = '2'
        elif resistancerange >= 20e-3 and resistancerange < 63.2e-3:
            resistancerangestring = '3'
        elif resistancerange >= 63.2e-3 and resistancerange < 200e-3:
            resistancerangestring = '4'
        elif resistancerange >= 200e-3 and resistancerange < 632e-3:
            resistancerangestring = '5'
        elif resistancerange >= 632e-3 and resistancerange < 2.0:
            resistancerangestring = '6'
        elif resistancerange >= 2.0 and resistancerange < 6.32:
            resistancerangestring = '7'
        elif resistancerange >= 6.32 and resistancerange < 20:
            resistancerangestring = '8'
        elif resistancerange >= 20 and resistancerange < 63.2:
            resistancerangestring = '9'
        elif resistancerange >= 63.2 and resistancerange < 200:
            resistancerangestring = '10'
        elif resistancerange >= 200 and resistancerange < 632:
            resistancerangestring = '11'
        elif resistancerange >= 632 and resistancerange < 2e3:
            resistancerangestring = '12'
        elif resistancerange >= 2e3 and resistancerange < 6.32e3:
            resistancerangestring = '13'
        elif resistancerange >= 6.32e3 and resistancerange < 20e3:
            resistancerangestring = '14'
        elif resistancerange >= 20e3 and resistancerange < 63.2e3:
            resistancerangestring = '15'
        elif resistancerange >= 63.2e3 and resistancerange < 200e3:
            resistancerangestring = '16'
        elif resistancerange >= 200e3 and resistancerange < 632e3:
            resistancerangestring = '17'
        elif resistancerange >= 632e3 and resistancerange < 2e6:
            resistancerangestring = '18'
        elif resistancerange >= 2e6 and resistancerange < 6.32e6:
            resistancerangestring = '19'
        elif resistancerange >= 6.32e6 and resistancerange < 20e6:
            resistancerangestring = '20'
        elif resistancerange >= 20e6 and resistancerange < 63.2e6:
            resistancerangestring = '21'
        elif resistancerange >= 63.2e6 and resistancerange < 200e6:
            resistancerangestring = '22'
        else:
            resistancerangestring = '1'

        #Send Resistance Range Command String
        commandstring = 'RDGRNG ' + str(channel) + ', ' + switchmode.get(mode,'1') + ', ' + exciterangestring + ',' + resistancerangestring + ',' + switchautorange.get(autorange,'0') + ', ' + switchexcitation.get(excitation,'0')
        self.write(commandstring)

    def GetHeaterStatus(self):

        switch = {
            '0' : 'no error',
            '1' : 'heater open error'
        }

        commandstring = 'HTRST?'
        result = self.ask(commandstring)
        status = switch.get(result, 'com error')

        return status

    def MagUpSetup(self, heater_resistance=1):
        ''' Setup the lakeshore for magup '''

        self.SetTemperatureControlSetup(channel=1, units='Kelvin', maxrange=10, delay=2, htrres=heater_resistance, output='current', filterread='unfiltered')
        self.SetControlMode(controlmode = 'open')
        self.SetControlPolarity(polarity = 'unipolar')
        self.SetHeaterRange(range=10)     # 1 Volt max input to Kepco for 100 Ohm shunt
        self.SetReadChannelSetup(channel = 1, mode = 'current', exciterange = 10e-9, resistancerange = 2e3,autorange = 'on')


    def DemagSetup(self, heater_resistance=1):
        ''' Setup the lakeshore for demag '''

        self.SetTemperatureControlSetup(channel=1, units='Kelvin', maxrange=10, delay=2, htrres=heater_resistance, output='current', filterread='unfiltered')
        self.SetControlMode(controlmode = 'open')
        self.SetControlPolarity(polarity = 'bipolar')  #Set to bipolar so that current can get to zero faster
        self.SetHeaterRange(range=10)  # 1 Volt max input to Kepco for 100 Ohm shunt
        self.SetReadChannelSetup(channel = 1, mode = 'current', exciterange = 10e-9, resistancerange = 2e3,autorange = 'on')


    def PIDSetup(self, heater_resistance=1):
        '''Setup the lakeshore for temperature regulation '''
        self.SetReadChannelSetup(channel=1, mode='current', exciterange=1e-8, resistancerange=63.2e3,autorange='on')
        sleep(15)  #Give time for range to settle, or servoing will fail
        self.SetReadChannelSetup(channel=1, mode='current', exciterange=1e-8, resistancerange=63.2e3,autorange='off')
        sleep(2)
        self.SetTemperatureControlSetup(channel=1, units='Kelvin', maxrange=100, delay=2, htrres=heater_resistance, output='current', filterread='unfiltered')
        self.SetControlMode(controlmode = 'closed')
        self.SetControlPolarity(polarity = 'unipolar')
        self.SetRamp(rampmode = 'off') #Turn off ramp mode to not to ramp setpoint down to aprox 0
        sleep(.5) #Give time for Set Ramp to take effect
        self.SetTemperatureSetPoint(setpoint=0.035)
        sleep(.5) #Give time for Setpoint to take effect
        self.SetRamp(rampmode = 'on' , ramprate = 0.2)
        self.SetHeaterRange(range=100) #Set heater range to 100mA to get 10V output range
        #self.SetReadChannelSetup(channel = 1, mode = 'current', exciterange = 1e-9, resistancerange = 2e3,autorange = 'on')

# Public Calibration Methods

    def SendStandardRuOxCalibration(self):
        pass

    def SendCalibration(self, filename, datacol, tempcol, curveindex, thermname='Cernox 1030', serialnumber='x0000', interp=True,\
                            temp_lim=300, tempco = 1, units=4):
        ''' Send a calibration based on a input file

            Input:
            filename: location of calibration file
            datacol: defines which column in filename will be used as data (zero indexed)
            tempcol: defines which column in filename will be used as temperature (zero indexed)
            curveindex: the curve index location in the lakeshore 370 (1 thru 20)
            thermname: sensor type
            serialnumber: thermometer serial number
            interp: if True the data will be evenly spaced from the max to min with 200 pts
                    if False the raw data is used.  User must ensure no doubles and < 200 pts
            temp_lim: temperature limit (K)
            tempco: 1 if dR/dT is negative, 2 if dR/dT is positive
            units: 3 to use ohm/K, 4 to use log ohm/K
        '''

        if curveindex < 1 or curveindex > 20:
            print ' 1 <= curveindex <= 20 for lakeshore 370'
            return 1

        #rawdata = read_array(filename) #obsolete, replace with genfromtxt
	rawdata = numpy.genfromtxt(filename)
        rawdatat = rawdata.transpose()
        datat = numpy.array(rawdatat[:,rawdatat[datacol,:].argsort()])

        #now remove doubles
        last = datat[datacol,-1]

        for i in range(len(datat[datacol,:])-2,-1,-1):
            if last == datat[1,i]:
                datat = numpy.hstack((datat[:,: i+1],datat[:,i+2 :]))
            else:
                last = datat[datacol,i]

        pylab.figure()
        pylab.plot(datat[datacol],datat[tempcol],'o')
        pylab.show()

        f = interp1d(datat[datacol],datat[tempcol])
        self.f = f

        # interpolate from min to max with 200 evenly spaced points if interp True
        if interp:
            Rs = scipy.linspace(min(datat[datacol]),max(datat[datacol]), num = 200)
        else:
            Rs = datat[datacol]
        Temps = f(Rs)

        pylab.figure()
        pylab.plot(datat[datacol],datat[tempcol],'o')
        #pylab.holdon()
        pylab.plot(Rs,Temps,'rx')
        pylab.show()

        # Send Header
        # 4, 350 ,1 -- logOhm/K, temperature limit, temperature coefficient 1=negative
        commandstring = 'CRVHDR ' + str(curveindex) + ', ' + thermname + ', ' + serialnumber + ', '+str(units)+', '+\
            str(temp_lim)+', '+ str(tempco)
        self.write(commandstring)
        print commandstring

        # Send Data Points
        for i in range(len(Rs)):
            pntindex = i+1
            if units == 4:
                logrofpoint = math.log10(Rs[i])
            else:
                logrofpoint = Rs[i]

            if Rs[i] < 10:
                stringlogrofpoint = '%(logrofpoint)7.5f' % vars()
            else:
                stringlogrofpoint = '%(logrofpoint)8.5f' % vars()

            tempofpoint = Temps[i]
            stringtempofpoint = '%(tempofpoint)5.3f' % vars()

            datapointstring = 'CRVPT ' + str(curveindex) + ', ' + str(pntindex) + ', ' + stringlogrofpoint + ', ' + stringtempofpoint
            self.write(datapointstring)
            print datapointstring

        pylab.figure()
        pylab.plot(Rs,Temps,'o')

    def SendMartinisRuOxCalibration(self, curveindex, thermname='RuOx Martinis', serialnumber='19740', interp=True,\
                            temp_lim=300, tempco = 1, units=4):
        self.SendMartinisCalibration(curveindex, thermname, serialnumber, interp, temp_lim, tempco, units)

    def SendMartinisCalibration(self, curveindex, thermname='RuOx Martinis', serialnumber='19740', interp=True,\
                            temp_lim=300, tempco = 1, units=4):
        ''' Send a calibration based on a input file

            Input:
            filename: location of calibration file
            datacol: defines which column in filename will be used as data (zero indexed)
            tempcol: defines which column in filename will be used as temperature (zero indexed)
            curveindex: the curve index location in the lakeshore 370 (1 thru 20)
            thermname: sensor type
            serialnumber: thermometer serial number
            interp: if True the data will be evenly spaced from the max to min with 200 pts
                    if False the raw data is used.  User must ensure no doubles and < 200 pts
            temp_lim: temperature limit (K)
            tempco: 1 if dR/dT is negative, 2 if dR/dT is positive
            units: 3 to use ohm/K, 4 to use log ohm/K
        '''

        if curveindex < 1 or curveindex > 20:
            print ' 1 <= curveindex <= 20 for lakeshore 370'
            return 1

        #rawdata = read_array(filename)
        #rawdatat = rawdata.transpose()
        #datat = numpy.array(rawdatat[:,rawdatat[datacol,:].argsort()])

        #now remove doubles
        #last = datat[datacol,-1]

        #for i in range(len(datat[datacol,:])-2,-1,-1):
        #    if last == datat[1,i]:
        #        datat = numpy.hstack((datat[:,: i+1],datat[:,i+2 :]))
        #    else:
        #        last = datat[datacol,i]

        #pylab.figure()
        #pylab.plot(datat[datacol],datat[tempcol],'o')
        #pylab.show()

        #f = interp1d(datat[datacol],datat[tempcol])
        #self.f = f

        # interpolate from min to max with 200 evenly spaced points if interp True
        #if interp:
        #Rs = scipy.linspace(min(datat[datacol]),max(datat[datacol]), num = 200)
        Rs = scipy.linspace(1258.92541, 63095.734448, num=200)
        #else:
        #    Rs = datat[datacol]
        Temps = (2.85 / (numpy.log((Rs-652.)/100.)))**4

        pylab.figure()
        #pylab.plot(datat[datacol],datat[tempcol],'o')
        #pylab.holdon()
        pylab.plot(Rs,Temps,'rx')
        pylab.show()

        # Send Header
        # 4, 350 ,1 -- logOhm/K, temperature limit, temperature coefficient 1=negative
        commandstring = 'CRVHDR ' + str(curveindex) + ', ' + thermname + ', ' + serialnumber + ', '+str(units)+', '+\
            str(temp_lim)+', '+ str(tempco)
        self.write(commandstring)
        print commandstring

        # Send Data Points
        for i in range(len(Rs)):
            pntindex = i+1
            logrofpoint = math.log10(Rs[i])

            if Rs[i] < 10:
                stringlogrofpoint = '%(logrofpoint)7.5f' % vars()
            else:
                stringlogrofpoint = '%(logrofpoint)8.5f' % vars()

            tempofpoint = Temps[i]
            stringtempofpoint = '%(tempofpoint)7.5f' % vars()

            datapointstring = 'CRVPT ' + str(curveindex) + ', ' + str(pntindex) + ', ' + stringlogrofpoint + ', ' + stringtempofpoint
            self.write(datapointstring)
            print datapointstring

        pylab.figure()
        pylab.plot(Rs,Temps,'o')
