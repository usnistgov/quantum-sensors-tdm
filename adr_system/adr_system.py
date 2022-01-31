'''
Created on Jan 11, 2010

@author: schimaf
'''
CONFIG_FILE     = "adr_system_setup.conf"
XML_CONFIG_FILE = "adr_system_setup.xml"

import os
from lxml import etree
import platform


from instruments import bluebox
from instruments import zaber

import numpy as np
import time
# I dont think anybody actually uses crate and tower thru AdrSystem anymore
# import crate
# import tower



class AdrSystem(object):
    '''
    Class to load all modules for controlling an ADR system. It calls crate, tower, and adrcontrol_mpl.
    This class should be used instead of directly calling those classes. It uses the adr_system_setup.conf
    file to define the local setup.

    The lakeshore370 class is only imported if one is on the system. This is because not all computers
    have GPIB installed and so importing was failing. This allows Mac OS X to run this class until python
    GPIB control is working.
    '''


    def __init__(self, app=None, lsync=40, number_mux_rows=8, dfb_settling_time=27,
    dfb_number_of_samples=4, doinit=False,
    logfolder = '/home/pcuser/data/ADRLogs/'):
        '''
        Constructor
        app = Can pass in an instance of pyqt4 Application class for realtime plotting
        lsync = line sync value that is being run
        number_mux_rows = Number of rows being muxed
        dfb_settling_time = Settling time in clock cycles.
        dfb_number_of_samples = Number of clocks cycles used to sample
        lsync > dfb_settling_time + dfb_number_of_samples + 1
        doinit = If true, send the serial commands to initialize the crate cards.
        '''

        self.settings_version = "1.0"

        # Name of the ADR system. e.g. Velma, Horton, Mystery Machine, Scooby, etc.
        self.app = app
        self.lsync = lsync
        self.number_mux_rows = number_mux_rows
        self.dfb_settling_time = dfb_settling_time
        self.dfb_number_of_samples = dfb_number_of_samples
        self.doinit = doinit
        self.name = ""
        self.temperature_controllers = []
        self.temperature_controller = None
        self.vmax = 0
        self.heater_resistance = 0
        self.heat_switches = []
        self.heat_switch = None
        self.gpib_instruments = []
        self.blueboxes = []
        self.logfolder = logfolder
        self.tccheckfolder = None
        self.magnet_controller = None
        self.measure_magnet_current = False
        self.measure_magnet_voltage = False
        self.magnet_control_relay = None

        # check that lsync -settlingtime -nsamp >= 1
        if not self.dfb_settling_time +self.dfb_number_of_samples < self.lsync - 1:
            print("WARNING: adr_system: dfb_settling_time + dfb_number_of_samples should be less than lsync - 1")
            # add this as a warning because it is still an issue under study - gco may 26 2011

        # Determine the config file path
        if platform.system() == 'Linux':
            self.config_file_path = '/etc/'
        elif platform.system() == 'Windows':
            # I don't know the best place for a win32 configuration file ...
            self.config_file_path = ''
        elif platform.system() == 'Darwin':
            self.config_file_path = '/etc/'

        #self.crate = crate.Crate(lsync=self.lsync, number_mux_rows=self.number_mux_rows, dfb_settling_time=self.dfb_settling_time, dfb_number_of_samples=self.dfb_number_of_samples, doinit=self.doinit)
        #self.crate.readConfigFile()
        #self.tower = tower.Tower()
        #tower_status = self.tower.readConfigFile()
        #if tower_status == False:
        #    self.tower = None

        status = self.readConfigFile()

        if status == True:
            try:
                import adrcontrol_mpl
                self.adr_control = adrcontrol_mpl.Adrcontrol(temperature_controller=self.temperature_controller,
                vmax=self.vmax, heat_switch=self.heat_switch, app=self.app,
                logfolder=self.logfolder)
            except:
                print("WARNING: adr_system.py failed to import adrcontrol_mpl")

    def readConfigFileXML(self, filename=None):
        '''
        Usage: self.readConfigFileXML(filename) - Reads the XML filename and imports
        the adr_system.
        '''
        if len(filename) > 0:
            #print("Actually loading tower XML file.")
            f = None
            root = None
            try:
                print(("Loading ADR system XML settings [%s]" % filename))
                f = open(filename, 'r')
                root = etree.parse(f)
            except:
                print("Could not load ADR system XML file.")

            if root is not None:
                child = root.find("name")
                if child is not None:
                    value = child.text
                    self.name = value

                child = root.find("max_voltage")
                if child is not None:
                    value = float(child.text)
                    self.vmax = value

                child = root.find("max_resistance")
                if child is not None:
                    value = float(child.text)
                    self.heater_resistance = value

                # Log file
                child = root.find("log_folder")
                if child is not None:
                    value = child.text
                    self.logfolder = value
                    print(("Setting log folder to [%s]" % self.logfolder))

                # TC Check file
                child = root.find("tc_check_folder")
                if child is not None:
                    value = child.text
                    self.tccheckfolder = value
                    print(("Setting Tc check folder to [%s]" % self.tccheckfolder))

                # Temperature Controllers
                print("Loading temperature controllers...")
                self.temperature_controllers = []
                child_temp_controllers = root.find("temperature_controllers")
                if child_temp_controllers is not None:
                    all_child_temp_controllers = child_temp_controllers.findall("temperature_controller")
                    for child_temp_controller in all_child_temp_controllers:
                        temperature_controller_type = child_temp_controller.get("type")
                        if temperature_controller_type == 'lakeshore370':
                            temperature_controller_gpib_address = int(child_temp_controller.get("gpib_pad"))
                            print(("Adding lakeshore 370 temperature controller (%i)." % temperature_controller_gpib_address))
                            # Only import the lakeshore class when needed because Mac OS X does not have the gpib class
                            from instruments import lakeshore370
                            ls370 = lakeshore370.Lakeshore370(pad=temperature_controller_gpib_address)
                            self.temperature_controller = ls370
                            self.temperature_controllers.append(ls370)
                        if temperature_controller_type == 'lakeshore370_serial':
                            port = child_temp_controller.get("port",default="lakeshore")
                            print(("Adding lakeshore 370 SERIAL temperature controller port = %s."%port))
                            from instruments import lakeshore370_serial
                            ls370 = lakeshore370_serial.Lakeshore370(port)
                            self.temperature_controller=ls370
                            self.temperature_controllers.append(ls370)

                # Heat switch
                print("Loading heat switches...")
                self.heat_switches = []
                child_heat_switches = root.find("heat_switches")
                if child_heat_switches is not None:
                    all_child_heat_switches = child_heat_switches.findall("heat_switch")
                    for child_heat_switch in all_child_heat_switches:
                        heat_switch_type = child_heat_switch.get("type")
                        if heat_switch_type == "zaber":
                            zaber_port = child_heat_switch.get("port")
                            print(("Adding %s heat switch. (port=%s)" % (heat_switch_type, zaber_port)))
                            a_heat_switch = zaber.Zaber(port=zaber_port)
                            a_heat_switch.SetHoldCurrent(0)
                            self.heat_switch = a_heat_switch
                            self.heat_switches.append(a_heat_switch)
#                        elif heat_switch_type == "hpdLabjack":
                        elif heat_switch_type == 'hpdLabjack':
                            print(("Adding %s heat switch." % (heat_switch_type)))
                            from instruments import heatswitchLabjack
                            self.heat_switch = heatswitchLabjack.HeatswitchLabjack()
                            self.heat_switches.append(self.heat_switch)
                        else:
                            print('No heat switch found.')
                        ######

                # MagnetRelay
                print("Loading magnet control relays...")
                self.magnet_control_relay=None
                child_relays = root.find("magnet_relays")
                if child_relays is not None:
                    all_child_relays = child_relays.findall("relay")
                    for child_relay in all_child_relays:
                        relay_type = child_relay.get("type")
                        print(relay_type)
                        if relay_type == "labjackU3_dougstyle":
                            from instruments import labjack
                            self.magnet_control_relay = labjack.Labjack()
                            print(("Adding %s as magnet_control_relay"%relay_type))
                        else:
                            print('No magnet control relay found.')


                # Vboxen
                print("Loading voltage boxes...")
                self.blueboxes = []
                child_voltage_boxes = root.find("blueboxes")
                if child_voltage_boxes is not None:
                    all_child_voltage_boxes = child_voltage_boxes.findall("bluebox")
                    for child_voltage_box in all_child_voltage_boxes:
                        voltage_box_type = child_voltage_box.get("version")
                        voltage_box_port = child_voltage_box.get("port")
                        print(("Adding %s voltage box. (port=%s)" % (voltage_box_type, voltage_box_port)))
                        a_voltage_box = bluebox.BlueBox(port=voltage_box_port, version=voltage_box_type)
                        self.blueboxes.append(a_voltage_box)

        print("Done reading ADR System XML config file.")

        return True

    def readConfigFile(self):
        print("Reading ADR system configuration...")
        # First try to open the XML config file
        xml_filename = self.config_file_path + XML_CONFIG_FILE
        if os.path.exists(xml_filename):
            print(("Found XML config file %s" % xml_filename))
            self.readConfigFileXML(xml_filename)
            return True
        print(("ADR system XML configuration file not found. [%s]" % xml_filename))

        # Try the old format
        filename = self.config_file_path + CONFIG_FILE
        if not os.path.exists(filename):
            print("ADR system configuration file not found.")
            return False

        print(("Found old config file %s" % filename))
        self.readConfigFileOld(filename)
        return True


    def readConfigFileOld(self, filename):
        '''
        readConfigFile() - Reads the configuration file in the default location which determines
        which instruments are on the local system.
        '''
        if filename is None:
            filename = self.config_file_path + CONFIG_FILE
        rcfile = open(filename, 'rt')
        print(("Reading adr system configuration [%s]" % filename))
        lines = rcfile.readlines()
        rcfile.close()#On module import, setup the list
        _namedports = {}
        for line in lines:
            if line[0] == '#' or len(line.strip()) <= 1:
                # Skip commented or blank lines
                continue
            field, value = line.split(":")
            value = value.strip()

            if field == "name":
                self.name = value
                print(("Name: %s" % self.name))
            elif field == "temperature_controller":
                tc_array = value.split(" ")
                temperature_controller_name = tc_array[0].strip()
                temperature_controller_gpib_address = int(tc_array[1].strip())
                if temperature_controller_name == 'lakeshore370':
                    print(("Adding lakeshore 370 temperature controller (%i)." % temperature_controller_gpib_address))
                    # Only import the lakeshore class when needed because Mac OS X does not have the gpib class
                    import lakeshore370
                    ls370 = lakeshore370.Lakeshore370(pad=temperature_controller_gpib_address)
                    self.temperature_controller = ls370
            elif field == "vmax":
                self.vmax = float(value)
                print(("Vmax = %f V" % self.vmax))
            elif field == "heater_resistance":
                self.heater_resistance = float(value)
                print(("heater resistance = %f ohm" % self.heater_resistance))
            elif field == "heat_switch":
                hs_array = value.split(" ")
                hs_type = hs_array[0].strip()
                hs_port = hs_array[1].strip()
                if hs_type == "zaber":
                    print(("Adding zaber heat switch. (port=%s)" % hs_port))
                    self.heat_switch = zaber.Zaber(port=hs_port)
                    self.heat_switch.SetHoldCurrent(0)
            elif field == "magnet_control":
                mc_array = value.split(" ")
                magnet_controller_name = mc_array[0].strip()
                magnet_controller_functionality = mc_array[1].strip()
                if magnet_controller_name == 'labjack':
                    print("Adding labjack for magnet controller.")
                    # Only import the labjack class when needed because it needs to be installed to not get an error
                    import labjack
                    lj = labjack.Labjack()
                    self.magnet_controller = lj
                if magnet_controller_functionality == 'ivrelay':
                    self.measure_magnet_current = True
                    self.measure_magnet_voltage = True
                    self.magnet_control_relay = True
                elif magnet_controller_functionality == 'irelay':
                    self.measure_magnet_current = True
                    self.measure_magnet_voltage = False
                    self.magnet_control_relay = True
                elif magnet_controller_functionality == 'relay':
                    self.measure_magnet_current = False
                    self.measure_magnet_voltage = False
                    self.magnet_control_relay = True
                elif magnet_controller_functionality == 'i':
                    self.measure_magnet_current = True
                    self.measure_magnet_voltage = False
                    self.magnet_control_relay = False
                else:
                    print('Did not recognize magnet controller functionality')
                    self.measure_magnet_current = False
                    self.measure_magnet_voltage = False
                    self.magnet_control_relay = False
            elif field == "bluebox":
                bb_array = value.split(" ")
                bluebox_port = bb_array[0].strip()
                bluebox_version = bb_array[1].strip()
                print(("Adding bluebox (%s, %s)" % (bluebox_port, bluebox_version)))
                bb = bluebox.BlueBox(port=bluebox_port, version=bluebox_version)
                self.blueboxes.append(bb)
            elif field == "gpib_instrument":
                gi_array = value.split("")
                gpib_instrument_type = gi_array[0].strip()
                gpib_instrument_address = gi_array[1].strip()
                if gpib_instrument_type == 'lakeshore370':
                    print(("Adding lakeshore 370 gpib instrument (%i)." % gpib_instrument_address))
                    ls370 = lakeshore370.Lakeshore370(pad=gpib_instrument_address)
                    self.gpib_instruments.append(ls370)
            elif field == "log_folder":
                self.logfolder = value
                print(("log folder = %s" % self.logfolder))
            elif field == "tccheck_folder":
                self.tccheckfolder = value
                print(("tccheck folder = %s" % self.tccheckfolder))

        print("Done reading legacy config file.")

    def displayConfig(self):
        '''
        displayConfig() - Display the instruments on the ADR system.
        '''
        print(("system name: %s" % self.name))
        if self.temperature_controller is not None:
            print(("Temperature controller: %s %s" % (self.temperature_controller.manufacturer, self.temperature_controller.model_number)))
        if self.vmax > 0:
            print(("Vmax = %f" % self.vmax))
        if self.heat_switch is not None:
            print("Heat switch: zaber")
        if len(self.blueboxes) > 0:
            for bb in self.blueboxes:
                print(("Bluebox %s %s" % (bb.version, bb.address)))
        if len(self.gpib_instruments) > 0:
            for instrument in self.gpib_instruments:
                print(("gpib instrument: %s %s" % (instrument.manufacturer, instrument.model_number)))

    def writeConfigFile(self, filename):
        '''
        writeConfigFile() - currently not implemented.
        '''
        pass

    def writeConfigFileXML(self, filename):
        '''
        writeConfigFileXML() - Output an XML configuration file.
        '''
        if filename == None:
            print("Please enter a filename.")
            return
        if (filename != []):
            if filename[-4:] == '.xml':
                savename = filename
            else:
                savename = filename + '.xml'

        f = open(savename, "w")

        # Create an xml object
        root = etree.Element("adr_config")

        root.set("settings_version", "%s" % self.settings_version)

        text = self.name
        child_name = etree.Element("name")
        child_name.text = text
        root.append(child_name)

        if self.vmax > 0:
            text = "%f" % self.vmax
            child_max_voltage = etree.Element("max_voltage")
            child_max_voltage.text = text
            root.append(child_max_voltage)

        if self.heater_resistance > 0:
            text = "%f" % self.heater_resistance
            child_max_resistance = etree.Element("max_resistance")
            child_max_resistance.text = text
            root.append(child_max_resistance)

        if self.logfolder is not None and len(self.logfolder) > 0:
            text = self.logfolder
            child_log_folder= etree.Element("log_folder")
            child_log_folder.text = text
            root.append(child_log_folder)

        if self.tccheckfolder is not None and len(self.tccheckfolder) > 0:
            text = self.tccheckfolder
            child_tc_check_folder= etree.Element("tc_check_folder")
            child_tc_check_folder.text = text
            root.append(child_tc_check_folder)

        if self.temperature_controller is not None:
            child_temp_controllers = etree.Element("temperature_controllers")

            if self.temperature_controller.manufacturer == "Lakeshore" and self.temperature_controller.model_number == "370":
                child_temp_controller = etree.Element("temperature_controller")
                child_temp_controller.set("type", "lakeshore370")
                child_temp_controller.set("gpib_pad", "%i" % self.temperature_controller.pad)
                child_temp_controllers.append(child_temp_controller)

            root.append(child_temp_controllers)

        # Heat switch
        if self.heat_switch is not None:
            child_heat_switches = etree.Element("heat_switches")
            child_heat_switch = etree.Element("heat_switch")
            if self.heat_switch.manufacturer == "Zaber":
                child_heat_switch.set("type", "zaber")
                child_heat_switch.set("port", self.heat_switch.port)
                child_heat_switches.append(child_heat_switch)
            root.append(child_heat_switches)

        # Blue boxes
        if len(self.blueboxes) > 0:
            child_blueboxes = etree.Element("blueboxes")
            root.append(child_blueboxes)
            for bluebox in self.blueboxes:
                child_bluebox = etree.Element("bluebox")
                child_bluebox.set("version", bluebox.version)
                child_bluebox.set("port", bluebox.port)
                child_blueboxes.append(child_bluebox)

        # GPIB instruments
        # not really working yet
        if len(self.gpib_instruments) > 0:
            child_gpib_instruments = etree.Element("gpib_instruments")
            root.append(child_gpib_instruments)
            for gpib_instrument in self.gpib_instruments:
                child_gpib_instrument = etree.Element("gpib_instrument")
                child_gpib_instrument.set("name", gpib_instrument.name)
                child_gpib_instrument.set("gpib_pad", bluebox.pad)
                child_gpib_instruments.append(child_gpib_instrument)

        # Save it to a file
        f.write(etree.tostring(root, pretty_print=True))

        f.close()

    def manualRampDownForWhenAdrGuiFails(self, target_duration_s=30*60, delay_s=5):
        hout_start = self.temperature_controller.getManualHeaterOut()
        nsteps = int(np.ceil(target_duration_s/delay_s))
        hout_per_step = hout_start/nsteps
        for i in range(nsteps):
            n_steps_left = nsteps-(i+1)
            new_hout = hout_start*(n_steps_left/nsteps)
            self.temperature_controller.setManualHeaterOut(new_hout)
            hout = self.temperature_controller.getManualHeaterOut()
            print(f"step {i} of {nsteps}, hout={new_hout:.2f}%, time left = {delay_s*n_steps_left:.2f} s")
            time.sleep(delay_s)