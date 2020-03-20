#!/usr/bin/env python

'''
extern_trig_client.py

Connects to a remote NDFB server and stores the external trigger bits (or changes in
them) to a file /path/to/file/ljhfilename.extern_trig whenever it detects that Matter
has started writing a new LJH data file. Stops when it detects that Matter has stopped
writing.

Created on Sept 19, 2014

@author: fowlerj
'''

import time, os, shutil
import optparse
import nasa_client
import numpy as np
import h5py
import bitarray

VERSION='0.9.2 beta'

class externTrigClient(object):
    """
    This client will connect to a server's external trigger pseudo-channels.

    It will write to a primary file (HDF5 format) and a backup file (plain binary
    data as stored by np.ndarray.tostring).  Both are stored with the latest data
    as given by the contents of ~/.daq/latest_ljh_pulse.cur (so make sure that
    directory exists, so matter can write to that file).

    The primary HDF5 file is closed and reopened every 5 minutes. While it's
    closed, it is backed up (and the prior backup is backed up), for safe keeping.
    """

    close_delay = 3 # seconds
    absurd_future = 1e99

    def __init__(self, host=None, port=None):
        self.client = nasa_client.Client(host=host, port=port, clockmhz=125)
        self.isStreaming = False
        self.client.stream_channels=()
        self.output_file = None
        self.output_filename = ""
        self.binary_file = None
        self.binary_filename = ""
        self.lastTrigValue = False
        self.n_triggers=0
        self.last_packet_frame = 0
        self.last_packet_rowcount = 0
        self.last_packet_timestamp_usec = 0
        self.last_trigger_rowcount = 0
        self.close_timer = self.absurd_future
        sentinel_filename = os.path.expanduser("~/.daq/latest_ljh_pulse.cur")
        self.sentinel_file = open(sentinel_filename, "r")
        print(('\n*** Starting extern_trig_client version %s ***'%VERSION))

    def check_sentinel_file(self):
        '''Open and close the output file as needed, based on the contents of the sentinel.'''
        self.sentinel_file.seek(0)
        words = self.sentinel_file.readlines()
        now = time.time()
        isopen = self.output_file is not None
        if isopen:
            if now > self.close_timer:
                self.close_timer = self.absurd_future
                return self.close_file()
            if len(words) < 1:
                return self.close_file()
            if len(words)>1 and 'closed' in words[-1]:
                if self.close_timer >= self.absurd_future:
                    self.close_timer = now + self.close_delay
                    print(('Closing output file in %d seconds.'%self.close_delay))
                return
            return

        # Output not open. Is that okay? If so, return.
        if len(words)<1 or (len(words)>1 and 'closed' in words[-1]):
            return

        # If we get here, then sentinel tells us we ought to be writing but we aren't.
        ljhname = words[0]
        parts = ljhname.split('.')
        basename = '.'.join(parts[:-1])
        basename += '_extern_trig.hdf5'
        self.open_file(basename)


    def open_file(self, filename):
        fp = h5py.File(filename, "a")
        self.output_file = fp
        self.output_filename = filename
        if fp is None: return

        print(("Opened output file '%s'"%filename))
        try:
            _ = fp['/trig_times']
            dataset_exists = True
        except KeyError:
            dataset_exists = False

        if dataset_exists:
            trig_rowcounts = fp['trig_times']
            assert trig_rowcounts.attrs['Nrows'] == self.client.nrow
            assert trig_rowcounts.attrs['sample_rate_hz'] == float(self.client.sample_rate)
        else:
            trig_rowcounts = fp.create_dataset('trig_times', (128,), dtype=np.uint64, maxshape=(None,),
                                       chunks=True, compression='gzip')
            trig_rowcounts.attrs['Nrows'] = self.client.nrow
            trig_rowcounts.attrs['Ntrigs'] = 0
            trig_rowcounts.attrs['Ncols'] = self.client.ncol
            trig_rowcounts.attrs['lsync'] = self.client.lsync
            trig_rowcounts.attrs['sample_rate_hz'] = float(self.client.sample_rate)

        self.binary_filename = os.path.splitext(filename)[0]+".dat"
        self.binary_file = open(self.binary_filename, "ab")

    def close_file(self):
        if self.output_file is not None:
            print(("Closing output file '%s'"%self.output_filename))
            trig_rowcounts = self.output_file['trig_times']
            nt = trig_rowcounts.attrs['Ntrigs']
            trig_rowcounts.resize((nt,))
            self.output_file.close()
            self.output_file = None
        if self.binary_file is not None:
            print(("Closing binary backup file '%s'"%self.binary_filename))
            self.binary_file.close()
            self.binary_file = None

    def backup_hdf5(self):
        if self.output_file is None: return
        filename = self.output_filename
        backup1_name = os.path.splitext(filename)[0]+"_backup1.hdf5"
        backup2_name = os.path.splitext(filename)[0]+"_backup2.hdf5"

        self.close_file()
        if os.path.isfile(backup1_name):
            shutil.copy2(backup1_name, backup2_name)
        shutil.copy2(filename, backup1_name)

        self.open_file(filename)

    def __del__(self):
        self.close_file()

    def connect_server(self, host=None, port=None):
        success = self.client.connect_server(host=host, port=port)
        if success:
            self.nrow = self.client.nrow
            self.sample_rate = self.client.sample_rate
            self.n_etrig_chan = (15+self.nrow)/16
            if self.n_etrig_chan > 4:
                raise NotImplementedError("The external trigger client can only handle 64 rows!")
            self.client.stream_channels = [i+self.client.CHANNEL_CODE_BASE_EXTERN_TRIG
                                           for i in range(self.n_etrig_chan)]
            self.chan_queues = tuple([[] for _ in range(self.n_etrig_chan)])

            self.dtype = np.uint16
            if self.nrow > 16:
                self.dtype = np.uint32
            if self.nrow > 32:
                self.dtype = np.uint64
            if self.nrow > 64:
                raise ValueError("%d rows > 64; cannot be used")
        return success

    def start_streaming(self): return self.client.start_streaming()
    def stop_streaming(self): return self.client.stop_streaming()
    def disconnect_server(self): return self.client.disconnect_server()

    def get_data_packets(self):
        payloads, headers = self.client.get_data_packets()
        for data, hdr in zip(payloads, headers):
            cnum = hdr['chan'] - self.client.CHANNEL_CODE_BASE_EXTERN_TRIG
            if hdr['packet_version'] <= 9:
                framecount = hdr['time_count_of_last_sample'] - hdr['record_samples'] + 1
            else:
                framecount = hdr['frame_count_of_last_sample'] - hdr['record_samples'] + 1
            # I think the timestamp also corresponds to the end of the packet, so correct to the front
            # timestamp is in usec, sample_rate is in inverse seconds, so I need to make units agree
            timestamp_usec = hdr['packet_timestamp']-1e6*hdr['record_samples']/float(self.client.sample_rate)
            assert cnum < self.n_etrig_chan
            self.chan_queues[cnum].append((data,framecount,timestamp_usec))
        return len(payloads)

    def process_data_packets(self):
        for q in self.chan_queues:
            if len(q) < 1:
                return

        self.check_sentinel_file()

        # If any queue has a packet older than any other queue's oldest, drop the older.
        newest_packet = np.max([q[0][1] for q in self.chan_queues])
        for i,q in enumerate(self.chan_queues):
            while q[0][1] < newest_packet:
                _data, t = q.pop(0)
                print(("Popping packet off queue %d at time %d"%(i, t)))

        nerr_flush = 0
        while True:
            shortest_queue = np.min([len(q) for q in self.chan_queues])
            if shortest_queue < 1:
                break

            # The following code relies on the idea that the packets from all of the extern
            # trigger pseudo-channels will have the same frame# boundaries. The assertions
            # are meant to verify this.
            alldata = [q.pop(0) for q in self.chan_queues]
            packlen = len(alldata[0][0])
            framestamp = alldata[0][1]
            timestamp_usec = alldata[0][2]
            rawbits = np.zeros(packlen, dtype=self.dtype)
            for i,a in enumerate(alldata[::-1]):
                data, this_framestamp, this_timestamp_usec = a
                assert packlen == len(data)
                assert this_framestamp == framestamp
                rawbits = (rawbits<<16) | np.asarray(data, dtype=np.uint16)

            bitvector = np2bitarray(rawbits, self.nrow, self.lastTrigValue)
            transitions = find_01_bit_transitions(bitvector)

            self.n_triggers += len(transitions)
            self.last_packet_frame = framestamp
            self.last_packet_rowcount = framestamp*self.nrow
            self.last_packet_timestamp_usec = timestamp_usec
            self.lastTrigValue = bitvector.pop()
            packet_trig_rowcounts = transitions + self.last_packet_rowcount
            if len(packet_trig_rowcounts)>0:
                self.last_trigger_rowcount = packet_trig_rowcounts[-1]

            if self.output_file is not None:
                trig_rowcounts = self.output_file['trig_times']

                # Resize the HDF5 dataset, if needed
                first = trig_rowcounts.attrs['Ntrigs']
                end = first + len(transitions)
                maxtrigs = int(trig_rowcounts.size)
                if end > maxtrigs:
                    maxtrigs = 128*((end+128)//128)
                    trig_rowcounts.resize((maxtrigs, ))

                trig_rowcounts[first:end] = packet_trig_rowcounts
                trig_rowcounts.attrs['Ntrigs'] = end

            if self.binary_file is not None:
                self.binary_file.write(packet_trig_rowcounts.tostring(order="C"))

        if self.binary_file is not None:
            self.binary_file.flush()
        try:
            if self.output_file is not None:
                self.output_file.flush()
                nerr_flush = 0
        except Exception as e:
            nerr_flush += 1
            if nerr_flush == 1 or nerr_flush%100 == 0:
                print(("** Failed to flush output file %d times in a row."%nerr_flush))
            if nerr_flush >= 1000000:
                print("** This is apparently a fatal problem.")
                raise e
        return


def np2bitarray(bitpattern, bitspervalue, startvalue=False):
    bitvector = bitarray.bitarray([startvalue], endian='little')

    bytesperstring = (7+bitspervalue)/8
    extrabits = 8*bytesperstring - bitspervalue

    if extrabits == 0:
        for bp in (bitpattern):
            bitvector.frombytes(bp.tostring()[:bytesperstring])
    else:
        for bp in (bitpattern):
            bitvector.frombytes(bp.tostring()[:bytesperstring])
            # Problem: we just added extrabits extra bits
            for _ in range(extrabits):
                bitvector.pop()
    return bitvector



def find_01_bit_transitions(bitvector):
    '''Given a vector of bits, return an array of sample numbers in which a 0->1
    bit transition occurred.  It is assumed that only the lowest <nbits> bits in
    each data value need to be checked.

    Here, "sample number" means row number. We're assuming that <nbits> is equal
    to the number of rows in the data stream. The sample numbers will be with respect
    to the first frame in the bitvector.
    '''
    # Subtract 1 because the bitarray was front-loaded with an initial value
    # (remembered from the previous search for transitions).
    return np.array(bitvector.search(bitarray.bitarray("01"))) - 1



def main(host='localhost', port='2011'):
    client = externTrigClient(host=host, port=port)
    client.connect_server()
    client.start_streaming()
    total_npackets=0
    last_alive_time = time.time()
    last_backup_time = time.time()
    last_ntrig = 0
    last_frame = 0
    ALIVE_DELAY = 5     # output message every 5 seconds.
    BACKUP_HDF5_PERIOD = 300  # back up the HDF5 file every 5 minutes.
    try:
        try:
            # Use Ctrl-C to break out of this infinite loop
            while True:
                now = time.time()
                if now > last_alive_time + ALIVE_DELAY:
                    ntrig = client.n_triggers - last_ntrig
                    dtime = float(client.last_packet_frame - last_frame)/client.sample_rate
                    try:
                        trig_rate = ntrig / dtime
                    except:
                        print("extern_trig_client tried to do ntrig / dtime with dtime = 0")
                    if last_frame <= 0:
                        trig_rate = 0.0
                    timestr = time.strftime("%x %X")
                    print(('%s. ExternTrigRate: %6g Hz. %6g packets seen. %s.' \
                        %(timestr, trig_rate, total_npackets, "Output off" if client.output_file is None else "Output on")))
                    last_ntrig = client.n_triggers
                    last_frame = client.last_packet_frame
                    last_alive_time = now

                if now > last_backup_time + BACKUP_HDF5_PERIOD:
                    client.backup_hdf5()
                    last_backup_time = now

                npackets = client.get_data_packets()
                j=0
                while npackets == 0:
                    time.sleep(0.1)
                    npackets = client.get_data_packets()
                    j+=1
                    if j%10==0: print("extern_trig_client not getting new packets?")
                client.process_data_packets()
                total_npackets += npackets
        except KeyboardInterrupt:
            print('*** Wrapping up extern_trig_client. ***')
        client.stop_streaming()
        client.disconnect_server()
    finally:
        client.close_file()


if __name__ == '__main__':
    p = optparse.OptionParser()
    p.add_option('-H','--host', action='store', dest='host', type='string',
                 help='Internet host name/address (default=localhost).')
    p.add_option('-p','--port', action='store', dest='port', type='string',
                 help='TCP port to connect to.')
    p.set_defaults(host='localhost')
    p.set_defaults(port='2011')
    opt, args = p.parse_args()
    host = opt.host
    port = opt.port
    main(host=host, port=port)
