# nist-qsp-tdm
This repo is intended to replace `nist_qsp_readout` and `nist_lab_internals` for the purposes of running TDM system under Python 3. Initially, it will contain only things that Galen needs to run various systems. Anything you need from the source repos can come in, as long as you test it and works with python 3 and PyQt5. I also strongly prefer to leave out all GPIB related things, but if you have something that is GPIB but lacks a serial port, we can include that code as well.

## cringe
GUI for setting up the crate and tower. Run as `cringe -L` from anywhere.

## adr_gui 
GUI for homebuilt cryo systems runs automags and a secondary feedback loop that keeps the thermometer temperature from drifting by changing the LS370 setpoint over time. Run as `adr_gui` from anywhere.


## Why bother with Python 3?
The last version of `matplotlib` that supports pytPythonhon 2.7 has a bug that makes `adr_gui` use up 100% of the system memory after a day or so. It seemed better to do something forward looking than to figure out which old version of `matplotlib` didn't have the bug.


# Before Install setup venv
```
sudo apt install python3-venv
python3 -m venv ~/qsp_venv
source ~/qsp_venv/bin/activate
pip install --upgrade pip
cd ~/qsp_venv
```

By switching to the `~/qsp_venv` folder we will cause all our code to be installed to `~/qsp_venv/src/pkg_name`.

# Installation
You probably want to install all the stuff you need for tdm, not just this repo, so
```  
pip install -e git+ssh://git@bitbucket.org/nist_microcal/nist-qsp-tdm.git#egg=nistqsptdm git+ssh://git@bitbucket.org/joe_fowler/mass.git@develop#egg=mass git+https://git@github.com/usnistgov/dastardcommander#egg=dastardcommander git+ssh://git@bitbucket.org/nist_microcal/realtime_gui.git#egg=realtime_gui
```
You may need to enter your bitbucket and or github login info. It may go smoother if you set up an ssh key with bitbucket and github so you don't need to enter loging info. If you have a a public ssh key setup, you may be required to add it. If you get an error like `Permission denied (public key)` you probably have to add your key to github or bitbucket (I've only seen this on github).

The code will be in `src` relative to where you run the install command. See the `dastardcommander` README for more info on pip installs. https://github.com/usnistgov/dastardcommander


You also need to copy `sudo cp namedserial/namedserialrc /etc` and `sudo cp adr_system/adr_system_setup.sample.xml /etc/adr_system_setup.xml` and update the entries therein to match your system.

You can figure out which of those to remove if you only need some.

Also install dastard and microscope:
  * https://github.com/usnistgov/microscope
  * https://github.com/usnistgov/dastard

You also need to install the `exodriver` (I'm pretty sure) for labjack, see instructions here: https://labjack.com/support/software/installers/exodriver
Or try the following (not it will create an exodriver directory, so do it somewhere out of the way or clean up)
```
sudo apt-get install build-essential libusb-1.0-0-dev install git-core
git clone git://github.com/labjack/exodriver.git
cd exodriver/
sudo ./install.sh
```

# Documentation
There is very little, but check in the `doc` folder and you might find some.

