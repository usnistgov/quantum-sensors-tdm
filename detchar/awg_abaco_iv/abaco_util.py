#!/usr/bin/env python3

import sys, time
import qsghw.core.helpers as corehelpers
import qsghw.core.interfaces as interfaces
import qsghw.core.packetparser as packetparser
import qsghw.fpgaip.helpers as fpgaiphelpers

(regaccess, debug, verbose) = (False, False, True)
(ctrlurl, dataurl) = ('pci://0000:17:00.0/fset1', 'udp://10.0.0.10')


(ctrlif, dataif, eids) = corehelpers.open(ctrlurl, dataurl, opendata=True, quiet=True) #verbose=False)
print(f"dataif {dataif}")

print("cmd =", " ".join(sys.argv))

fset = fpgaiphelpers.makeinstance(ctrlif, eids, ":fullset:", verbose=debug)
tonegen = fpgaiphelpers.makeinstance(ctrlif, eids, ":tonegen:", verbose=debug)
frontend = fpgaiphelpers.makeinstance(ctrlif, eids, interfaces.channelizer, verbose=debug)
backend = fpgaiphelpers.makeinstance(ctrlif, eids, ":backend:", verbose=debug)
assert fset and tonegen and frontend and backend
fset.debug = tonegen.debug = frontend.debug = backend.debug = (regaccess, debug, verbose)

# route sync signal (the settings below are the default)
fset.write_sync4sel(0) # external sync #0 signal to suncbus #4
fset.write_sync4mode(0) # pass through
fset.write_sync5sel(5) # tonegen's sync-Q signal (syncsrc bit #5) to syncbus #5
fset.write_sync5mode(0) # pass through

# backend output selection: disable normal output, enable ATAN1 on monitor output
backend.write_pktrst(1)
backend.write_pktsrc("None", "pktdata")
backend.write_pktrst(0)

backend.write_monrst(1)
backend.write_pktsrc("ATAN1", "monitor")
backend.write_monmerge(14)
backend.write_mondec(1)
backend.write_monrst(0)

# set up module signal generation: sine wave on tone generator's 2nd ('Q') output
samplerate = tonegen.samplerate = frontend.samplerate
iqorder = tonegen.get_iqorder()
print("samplerate = %8.3f MHz   iqorder = %d   %s" % (samplerate/1e6, iqorder, tonegen))


freq_hz = 10
tonegen.set_q(freq=freq_hz, amp=1, phase=0., mode="square-q", duty=0.25)
        #tonegen.dumpfields()

def getdata(toread, tosleep_s):

    time.sleep(tosleep_s)
    dataif.reset()
    # time.sleep(tosleep_s)
    (nbytes, buffer) = dataif.readall(toread)
    # print(nbytes, buffer[0:31])

    (headeroffsets, payloadoffsets, payloadlengths, seqnos) = packetparser.findlongestcontinuous(buffer, incr=1024)
    (consumed, alldata) = packetparser.parsemany(buffer, headeroffsets[0], payloadoffsets, verbose = True)
    # print("longest continuous:", toread, "->", nbytes, "->", consumed, "=", len(payloadoffsets), "packets =", sum(payloadlengths), "byte total payload")
    # print(alldata.data.shape, "x", alldata.data.dtype)
    return alldata

#get sync signal and unwrap + convert atan
# sync = alldata.data['syncbus'].astype(int) & 32
# sync_in = alldata.data['syncbus'].astype(int) & 16
# atan1 = np.unwrap(2*np.pi*alldata.data['scal2']/2**32, axis=0)
# print(sync)
# print(atan1)
        
        
# dataif.close()
# ctrlif.close()

# ch=0
# plt.figure()
# plt.plot(atan1[:,ch])
# plt.ylabel(f"atan1 ch {ch}")
# plt.xlabel("sample number")

# plt.figure()
# plt.plot(sync_in[:, ch])
# plt.ylabel(f"sync_in ch {ch}")
# plt.xlabel("sample number")