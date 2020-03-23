# nist-qsp-tdm
This repo is intended to replace `nist_qsp_readout` and `nist_lab_internals` for the purposes of running TDM system under Python 3. Initially, it will contain only things that Galen needs to run various systems. Anything you need from the source repos can come in, as long as you test it and works with python 3 and PyQt5. I also strongly prefer to leave out all GPIB related things, but if you have something that is GPIB but lacks a serial port, we can include that code as well.

## cringe
GUI for setting up the crate and tower. Run as `cringe -L` from anywhere. Save cringe setup files in `~/cringe_config`.

## adr_gui 
GUI for homebuilt cryo systems runs automags and a secondary feedback loop that keeps the thermometer temperature from drifting by changing the LS370 setpoint over time. Run as `adr_gui` from anywhere.

## tower_power_gui
GUI for turning on/off the tower power supplies in the right order so it doesn't blow up the power card. Run as `tower_power_gui`.

## Why bother with Python 3?
The last version of `matplotlib` that supports pytPythonhon 2.7 has a bug that makes `adr_gui` use up 100% of the system memory after a day or so. It seemed better to do something forward looking than to figure out which old version of `matplotlib` didn't have the bug.


# Before Install setup venv
```
sudo apt install python3-venv
```
Then you can copy and paste the entire next block hopefully.
```
python3 -m venv ~/qsp
source ~/qsp/bin/activate
pip install --upgrade pip
echo "source ~/qsp/bin/activate" >> ~/.bashrc
```

The echo line adds `source ~/qsp/bin/activate` to `~/.bashrc`, which activates this venv for each new terminal. You may exit this venv with `deactivate` to run code that requires a different python environment.

# Installation
You probably want to install all the stuff you need for tdm, not just this repo, so
```  
pip install -e git+ssh://git@bitbucket.org/nist_microcal/nist-qsp-tdm.git#egg=nistqsptdm -e git+ssh://git@bitbucket.org/joe_fowler/mass.git@develop#egg=mass -e git+https://git@github.com/usnistgov/dastardcommander#egg=dastardcommander -e git+ssh://git@bitbucket.org/nist_microcal/realtime_gui.git#egg=realtime_gui
```
You may need to enter your bitbucket and or github login info. It may go smoother if you set up an ssh key with bitbucket and github so you don't need to enter loging info. If you have a a public ssh key setup, you may be required to add it. If you get an error like `Permission denied (public key)` you probably have to add your key to github or bitbucket (I've only seen this on github).

Here we install with the `-e` "editable" flag. On Ubuntu 18 in the venv this causes the code to be installed to `qsp/src`. On a Mac without a venv I found the code to be installed to `src` relative to where the command was executed.

You also need to copy `sudo cp namedserial/namedserialrc /etc` and `sudo cp adr_system/adr_system_setup.sample.xml /etc/adr_system_setup.xml` and update the entries therein to match your system. There are additonal example files in `doc/etcfiles`.

You can figure out which of those to remove if you only need some.

Also install dastard and microscope:
  * https://github.com/usnistgov/microscope
  * https://github.com/usnistgov/dastard

You also need to install the `exodriver` (I'm pretty sure) for labjack, see instructions here: https://labjack.com/support/software/installers/exodriver
Or try the following:
```
sudo apt-get install build-essential libusb-1.0-0-dev install git-core
git clone git://github.com/labjack/exodriver.git
cd exodriver/
sudo ./install.sh
cd ..
rm -r exodriver
```

# Documentation
There is very little, but check in the `doc` folder and you might find some.

