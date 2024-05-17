# quantum-sensors-tdm
This repo is intended to replace [`nist_qsp_readout`](https://bitbucket.org/nist_microcal/nist_qsp_readout) and [`nist_lab_internals`](https://bitbucket.org/nist_microcal/nist_lab_internals) for the purposes of running TDM system under Python 3. Initially, it contained only things that Galen needs to run various systems. Anything you need from the source repos can come in, as long as you test it and works with Python 3 and PyQt5. I also strongly prefer to leave out all GPIB related things, but if you have something that is GPIB but lacks a serial port, we can include that code as well.

## Cringe
GUI for setting up the crate and tower. Run as `cringe -L` from anywhere. Save cringe setup files in `~/cringe_config`.

## ADR GUi
GUI for homebuilt cryo systems runs automags and a secondary feedback loop that keeps the thermometer temperature from drifting by changing the LS370 setpoint over time. Run as `adr_gui` from anywhere.

## tower_power_gui
GUI for turning on/off the tower power supplies in the right order so it doesn't blow up the power card. Run as `tower_power_gui`.

## tdm_term
Run `tdm_term` to launch a terminal with named tabs and running dastard, cringe and dastard commander. Requires dastard and dastard commander to be installed. The first time you open roxterm (via `tdm_term`) go to Preferenes->Edit Current Profile->General and change "When command exists:" to "Restart Command". This seems to cause it to leave the terminal open for new commands.

# Before installation, set up venv
Install python3-venv and roxterm. For certain versions of Ubuntu (17.10 and newer), official Debian packages of roxterm are unavaible. An unsupported version of roxterm can be installed by adding the PPA described at https://launchpad.net/~h-realh/+archive/ubuntu/roxterm prior to running the apt install commands.
```
sudo add-apt-repository ppa:h-realh/roxterm
sudo apt-get update
sudo apt install python3-dev python3-venv roxterm
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
You probably want to install all the stuff you need for TDM, not just this repo, so
```  
pip install -e git+ssh://git@github.com/usnistgov/quantum-sensors-tdm.git#egg=quantumsensorstdm
pip install -e git+ssh://git@github.com/usnistgov/mass.git#egg=mass
pip install -e git+https://git@github.com/usnistgov/dastardcommander#egg=dastardcommander
```
You may need to enter your github login info. It may go smoother if you set up an ssh key with github so you don't need to enter loging info. If you have a a public ssh key setup, you may be required to add it. If you get an error like `Permission denied (public key)` you probably have to add your key to github.

Here we install with the `-e` "editable" flag. On Ubuntu 18 in the venv this causes the code to be installed to `qsp/src`. On a Mac without a venv I found the code to be installed to `src` relative to where the command was executed.

You also need to copy `sudo cp namedserial/namedserialrc /etc` and `sudo cp adr_system/adr_system_setup.sample.xml /etc/adr_system_setup.xml` and update the entries therein to match your system. There are additonal example files in `doc/etcfiles`.

You can figure out which of those to remove if you only need some.

Also install the non-Python packages, Dastard and Microscope:
  * https://github.com/usnistgov/dastard
  * https://github.com/usnistgov/microscope


You also need to install the `exodriver` (I'm pretty sure) for labjack, see instructions here: https://labjack.com/support/software/installers/exodriver
Or try the following:
```
cd ~/Downloads
sudo apt-get install build-essential libusb-1.0-0-dev install git-core
git clone git://github.com/labjack/exodriver.git
cd exodriver/
sudo ./install.sh
cd ..
rm -r exodriver
```
On ubuntu 20 use this instead:
```
sudo apt-get install build-essential libusb-1.0-0-dev  git-core
```
# Documentation
There is very little, but check in the `doc` folder and you might find some.

# Cringe External Commands

  * Cringe has a ZMQ Reply (you use as Request) socket listening on port 5509. It accepts the commands;
    * `SETUP_CRATE`
    * `FULL_TUNE`
  * See tests `tests/test_cringe.py` for more info on replies. 

# Development tips  

## Working on cringe
  
  * Make a `/etc/namedserialrc` file with `dummy rack` and `dummy crate`
  * `cringe -A 0 1 2  -S 0 1 2 -C DFBCLK DFBx2 BAD16 -T DB1 13` will launch cringe with one of each useful card
  * `-D` flag will turn on debug output
  * `pytest -s` is useful to see all output, `pytest-watch` will hang due some unknown issue with the `zmq_rep` tests, so just use `pytest`
