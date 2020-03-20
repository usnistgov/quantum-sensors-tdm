# Autotune in Cringe
Galen O'Neil, July 27 2016

Autotune in Cringe is in an early state of development. Many choices have been hardcoded to work with the two systems NIST has deployed using Cringe. Please report any problems by filing an issue at the `nist_lab_internals` repo on BitBucket. Before reporting, make sure the issue still exists in the most recent commit in `nist_lab_internals` master branch.

Cringe works by having gui calls translate directly into communication with the crate, this helps maintain consitency between the gui and the internal state of the crate. Autotune works by making direct calls to Qt to manilpulate the Cringe gui, so it should leave in cringe in a state that accuratley reflects the internal state of the crate. 

## Full Tune
When you press the full tune button, the following occurs
1. Take an FBB vphi.
2. Use the results of 1 to lock FBB.
3. Take a locked FBA (squid 1) vphi.
4. Use the results of 3 to choose FBB offset. 
    * FBB offset is chosen to center the squid 1 vphi on the upward going slope of the FBB vphi. This is hardcoded, and I'm about 90% sure it is the upward one.
5. Take unlocked FBA (squid 1) vphi.
6. Use the results of 5 to choose ADC (error offset), FBA offset, P, I and Mix.
    1. Choose ADC offset to satisfy the "Lockpoint % from bottom of vphi" condition.
    2. Choose lowest value of FBA such that error will be nearly zero with feedback off. There is a hardcoded choice of minimum value of FBA related to the vphi period.
    3. Calculate the slope at the lockpoint.
    4. I = (I*Slope product)/slope as an integer
    5. Mix = (Mix*Slope product)/slope
        * The Mix is not sent to the server. Instead it is written to `~/nasa_daq/matter/autotune_mix.mixing_config`. Load that file with MATTER.
7. Lock FBA, turn on ARL.
8. Many plots are now on screen. The topmost plot shows the unlocked squid1 vphis, shifted along the x-axis to lie on top of each other.

## Assumptions
1. nist_qsp_readout has had `setup.py install` run
2. Server Column 0 is associated with first DFBx2 tab channel 1, Server Column 1 is associated with first DFBx2 tab channel 2, Server Column 2 is associated with second DFBx2 tab channel 1.
3. All Tower DAC values have been tuned, TESs have been biased.
    * The shock button next to tower cards is intended for biasing TESs. For each channel it turns the output to max, waits a few ms, then turns the output back to the value in the gui.
4. System and Class Globals have been tuned, as well as any other things you care about.

## Other Features
* Check the info next to the "startclient" button. It should be right. You shouldn't need the press the button, it should just refresh every few seconds. 
* If you start the bit error test, you need to press the stop button. Closing the window that appears will not stop the test. You should see zero errors in the test, and nice narrow histogramsin the plots. 
* The Settling time sweeps are useful for tuning SETT, NSAMP and DFB prop delay. They are run by first choosing values of FBA that cause the error signal to to a low value fo even rows and a high value for odd rows. Then NSAMP is set to 1, and SETT is sweeped through all valid values. Error vs SETT is plotted for each row.
* The buttons in the bottom near "Squid bias sweeper" should make plots of svphi amplitude vs the thing in the button title.
* FBA goes to Squid1 feedback. FBB goes to Series Array Feedback, or Squid 2 feedback if you have it. 
* There is a plot noise button that will plot noise PSDs for each row.

## Frequently Asked Questions
Q: Do I need to be streaming the full number of columns specified in cringe? ie. My crate has 4x dfbx2’s and I have cringe set-up for 4, but right now I am only specifying the fibers for one column on the sever settings (since I only have 1 column of squids).
A: No, you can use fewer columns, but if you only use one column you must use Channel 1 of the first DFBx2. If you only use two columns, you must use both channels of the first DFBx2. 

Q: How does it handle rows which may be bad, open RA say (I have a few of those)? 
A: Currently it has no special handling for bad rows. So they will have essentialy random values of FBA, FBB, I and Mix. It's actualy pretty easy to programatically identify bad rows and simply turn them off (I=0, Mix=0), but this isn't implemented. 