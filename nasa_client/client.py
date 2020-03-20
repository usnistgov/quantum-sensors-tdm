'''
client.py

Classes to connect to the NASA ndfb_server for reading out NIST multiplexed detectors.

Use one of:
1) ZMQClient() for server version 3.3.0+, using ZMQ connections
2) TCPClient() for server versions <3.3.0, using raw TCP connections
3) Client() factory function to automatically choose by trying the old then the new kind.

Created on Sep 22, 2011
Adapted to ZMQ connections June 2015

@author: fowlerj
'''
import numpy
import socket
import struct

from . import network_pipe
from . import xcaldaq_commands
import zmq

class NoMorePackets(StopIteration): pass

class NDFBClient(object):
    """
    An object to serve as the client of a NASA ndfb_server.

    This is a an abstract base class. For servers version 3.3.0+, when the connections use
    ZMQ, use ZMQClient. For pre-3.3.0, use the TCPClient object, or if unsure, use the 
    client.Client() factory function.
    """

    DEFAULT_PORT = 60000
    CHANNEL_CODE_ERR_SUMMARY = 999990
    CHANNEL_CODE_FB_SUMMARY = 999991
    CHANNEL_CODE_ERR_STDDEV = 999992
    CHANNEL_CODE_FB_STDDEV = 999993
    CHANNEL_CODE_BASE_EXTERN_TRIG = 99800

    def __init__(self):
        self.connected = False
        self.streaming = False

        self.ncol = 0
        self.nrow = 0
        self.npixels = 0
        self.lsync = -1  # in units of master clock ticks per pixel
        self.nchan = -1

        self.curdatapacket = 0
        self.curpix = 0
        self.decimate_level = 10
        self.partial_packet = None

        # This is a list of the desired stream numbers to actually stream
        self.stream_channels = []

    def get(self, secondary, channel):
        """Get a parameter selected by <secondary> for the given <channel> (for some
        parameters, <channel> is ignored.

        <secondary> can be a code number (see xcaldaq_commads.secondary_comm for a list)
        or a string, which will be looked up from that list (actually, a python dict).

        If that particular parameter is a float, you should call .get_float() instead."""
        if isinstance(secondary, str):
            secondary = xcaldaq_commands.secondary_comm[secondary]
        return self._command(xcaldaq_commands.primary_comm['GET'], secondary, channel, 0)

    def get_float(self, secondary, channel):
        """Get a floating-point parameter selected by <secondary>. If that particular parameter
        is not a float you should call .get() instead."""
        raw = self.get(secondary, channel)
        return struct.unpack("f", struct.pack("I", raw))

    def setp(self, secondary, channel, level):
        if isinstance(secondary, str):
            secondary = xcaldaq_commands.secondary_comm[secondary]
        return self._command(xcaldaq_commands.primary_comm['SET'], secondary, channel, level)

    def set_float(self, secondary, channel, level):
        packed_level = struct.unpack("I", struct.pack("f", level))[0]
        raw = self.setp(secondary, channel, packed_level)
        return struct.unpack("f", struct.pack("I", raw))

    def _command(self, primary, secondary, channel, level):
        """
        Send a command to the server and await a reply.
        The command language is defined in XCALDAQCommands.h, which can be found in either of the NASA
        programs xcaldaq_client and ndfb_server.

        <primary>    A 2-byte primary command, defined in xcaldaq_commands.primary_comm
        <secondary>  A 2-byte secondary command, defined in xcaldaq_commands.secondary_comm
        <channel>    A 4-byte channel number (because we really need to have 4 billion channels)
        <level>      A 4-byte parameter to the primary/secondary command.

        Generally, primary/secondary will mean something like SET/X_PARAMETER or GET/Y_PARAMETER,
        and level will then be the value of the parameter to be set or read.

        Returns: a 32-bit value whose meaning depends on the primary/secondary command.
        """
        NETWORK_ORDER = '>'
        USHORT = 'H'
        PADDING = 'x'
        ULONG = 'L'
        FMT = NETWORK_ORDER + 3 * USHORT + 2 * PADDING + 2 * ULONG

        # Warning!  These two "padding" bytes are nominally an error flag then an error number.
        # As the xcaldaq_client code ignores them, we shall do so too....for now.
        msg = struct.pack(FMT, xcaldaq_commands.comm_ack['COMMAND'], primary, secondary, channel, level)

        # With the comm pipe locked, send the command, and get the reply.
        self.commPort.send(msg)
        poller = zmq.Poller()
        poller.register(self.commPort, zmq.POLLIN)
        if poller.poll(1000):
            reply = self.comm_recv()
        else:
            raise IOError("aint got no response from the server on comm port")

        if len(reply) < 16:
            print('Server not responding, probably crashed.')
            return -1

        # Parse the reply
        (rcmd, rprimary, rsecondary, rchannel, rlevel) = struct.unpack(FMT, reply)

        # Verify that this reply matches the primary/secondary/channel used in the command.
        for (testname, found, expected) in zip(('ACK', 'primary', 'secondary', 'channel'),
                                               (rcmd, rprimary, rsecondary, rchannel),
                                               (xcaldaq_commands.comm_ack['ACKNOWLEDGE'],
                                                primary, secondary, channel)):
            if found != expected:
                raise IOError("Server reply packet had %s=%d, but expected %d" % (
                               testname, found, expected))

        if self.debug is True:
            print(('_command %s, %s, %s, %s, %s, %s, %s, %s' % (FMT, primary, secondary, rcmd,
                                                             rprimary, rsecondary, rchannel, rlevel)))
        return rlevel

    def clear_data_queue_by_getting_all_packets(self):
        print(('cleared data queue, threw away %d packets' % len(self._get_raw_packets())))

    def clear_data_queue_by_stop_start_streaming(self):
        self.stop_streaming()
        self.start_streaming()

    def packet_payload_to_data(self, payload, numpy_data_type):
        return numpy.fromstring(payload, dtype=numpy_data_type)

    def print_server_info(self):
        print(('Lsync = %d clock ticks/pixel, nCol = %d, nRow = %d, nSamples = %d, sampleRate = %d' % (
                        self.lsync, self.ncol, self.nrow, self.num_of_samples, self.sample_rate)))

    def _subclassCallback(self, callbackName, *args, **kwargs):
        try:
            command = 'self.%s(*args, **kwargs)' % callbackName
            exec(command)
        except AttributeError:
            pass

    def get_data_packets(self, max_bytes = 10000000):
        raw_packets = self._get_raw_packets(max_bytes)
        headers = []
        data = []
        for rp in raw_packets:
            this_header = self.parse_packet_header(rp)
            header_size = this_header['header_bytes']
            headers.append(this_header)
            data_array = self.packet_payload_to_data(rp[header_size:],
                                                     this_header['numpy_data_type'])
            data.append(data_array)
        return data, headers

    def __delattr__(self, *args, **kwargs):
        self.disconnect_server()
        return object.__delattr__(self, *args, **kwargs)

    def parse_packet_header(self, pkt):
        """Parse the header of an ndfb_server packet <pkt> and return a dictionary
        containing the relevant parameters.
        """
        # The header has a consistent format and size
        # x = 1-byte padding (ignore)
        # B = 1-byte unsigned char
        # H = 2-byte unsigned short
        # I = 4-byte unsigned int
        # Q = 8-byte unsigned long long

        version_fmt = "QQBB"
        version_size = struct.calcsize(version_fmt)
        if len(pkt) < version_size:
            raise ValueError("Packet size %d is too short to parse the header first half (min size %d)." % 
                             (len(pkt), version_size))

        _, _, _, version = struct.unpack(version_fmt, pkt[:version_size])

        if version == 10:
            d = _parse_packet_header_v10(pkt)
        elif version >= 5 and version <= 9:
            d = _parse_packet_header_v56789(pkt)
        # We only know how to parse packet version 10.  If necessary, this could be generalized.
        else:
            raise NotImplementedError("This packet header claims to be from version %d.  "
                                      "Only versions 5-10 (inclusive) can be parsed" % 
                                      version)

        SIGNED_SAMPLES_FLAG = 0x02
        NETWORK_PACKET_ORDER_FLAG = 0x40
        if version == 7 or (d['flags'] & NETWORK_PACKET_ORDER_FLAG):
            self.network_order = True
            numpy_data_type = ">"
        else:
            self.network_order = False
            numpy_data_type = "<"
        if (d['flags'] & SIGNED_SAMPLES_FLAG):
            numpy_data_type += "i2"
        else:
            numpy_data_type += "u2"
        d['numpy_data_type'] = numpy_data_type
        return d    



def _parse_packet_header_v56789(pkt):
    header1_fmt = "!IIIIBBHHxx"
    header1_size = struct.calcsize(header1_fmt)

    if len(pkt) < header1_size:
        raise ValueError("Packet size %d is too short to parse the header first half (min size %d)." % 
                         (len(pkt), header1_size))

    d = {}
    params1 = struct.unpack(header1_fmt, pkt[:header1_size])
    names1 = ('chan', 'size_bytes', 'header_bytes', 'record_samples', 'bits_per_samp',
             'packet_version', 'flags', 'decimation_level')
    for name, p in zip(names1, params1):
        d[name] = p

    if len(pkt) < d['header_bytes']:
        raise ValueError("Packet size %d is too short to parse the header second half (min size %d)." % 
                         (len(pkt), d['header_bytes']))

    # Now we know version number, we can be sure about the rest
    v = d['packet_version']
    names2 = ['count_of_last_sample', 'time_when_server_started_usec_since_epoch',
            'mix_ratio', 'sample_rate_numerator', 'sample_rate_denominator',
             'volt_offset', 'volt_scale', 'time_count_of_last_sample',
             'max_raw', 'min_raw', 'packet_timestamp']
    if v in (5, 6):
        header2_fmt = "!QQfxxxxIIffQ"
    elif v in (7, 8):
        header2_fmt = "!QQfxxxxIIffQII"
    elif v in (9,):
        header2_fmt = '!QQfxxxxIIffQIIQ'
    params2 = struct.unpack(header2_fmt, pkt[header1_size:d['header_bytes']])
    for name, p in zip(names2, params2):
        d[name] = p
    return d



def _parse_packet_header_v10(pkt):
    # The header has a consistent format and size
    # x = 1-byte padding (ignore)
    # B = 1-byte unsigned char
    # H = 2-byte unsigned short
    # I = 4-byte unsigned int
    # i = 4-byte signed int
    # Q = 8-byte unsigned long long
    header_fmt = "<IIIIBBHIIfffiiQQQQ"
    header_size = struct.calcsize(header_fmt)

    if len(pkt) < header_size:
        raise ValueError("Packet size %d is too short to parse the header (min size %d)." % 
                         (len(pkt), header_size))

    d = {}
    params1 = struct.unpack(header_fmt, pkt[:header_size])
    names1 = ('chan', 'record_samples', 'header_bytes', 'flags',
            'bits_per_samp', 'packet_version', 'decimation_level',
            'sample_rate_numerator', 'sample_rate_denominator', 'mix_ratio',
            'volt_offset', 'volt_scale', 'max_raw', 'min_raw',
            'count_of_last_sample', 'time_when_server_started_usec_since_epoch',
            'packet_timestamp', 'frame_count_of_last_sample')
    for name, p in zip(names1, params1):
        d[name] = p
    d['size_bytes'] = d['header_bytes'] + (d['bits_per_samp'] / 8) * d['record_samples']
    return d



class ZMQClient(NDFBClient):
    """
    An object to serve as the client of a NASA ndfb_server.

    This is a client for servers version 3.3.0+, when the connections use
    ZMQ. For pre-3.3.0, use the TCPClient object, or if unsure, use the 
    client.Client() factory function.
    """


    def __init__(self, host=None, port=None, debug=False, clockmhz=50, noblock=True):
        """
        <host>   The host name or IP address for the ndfb_server.
        <port>   The port number to which you'll connect (if None, then self.DEFAULT_PORT is used).

        If <host> is None, then it must be filled in when calling connect(...)
        """
        NDFBClient.__init__(self)
        self.context = zmq.Context()
        self.host = host
        if port is None:
            self.port = self.DEFAULT_PORT
        else:
            self.port = int(port)
        self.debug = debug  # prints out additional debugging statements when this is true

        self.commPort = self.context.socket(zmq.REQ)
        self.dataPort = self.context.socket(zmq.SUB)
        self.commPort.setsockopt(zmq.LINGER,0)
        self.dataPort.setsockopt(zmq.LINGER,0)
        self.dataPort.setsockopt(zmq.RCVTIMEO,10000)
        self.clockhz=clockmhz*1000000
        self.noblock=noblock

    def __subclassCallback(self, callbackName, *args, **kwargs):
        try:
            command = 'self.%s(*args, **kwargs)' % callbackName
            exec(command)
        except AttributeError:
            pass

    def connect_server(self, host=None, port=None):
        """
        Connect to an ndfb_server at Internet location <host>:<port>.
        If either <host> or <port> is None, the previous value will be used.  See the constructor
        for how the initial "previous" values are set.

        Internally, this method must connect two sockets: one for commands/replies, and the other
        for data.  (Data is one-way, in that packets cannot be sent from here to the server
        on the data pipe.) These use the nominal TCP port P, and also P+1, respectively.
        """
        if self.connected:
            print('Already connected')
            return

        if host is None and self.host is None:
            raise ValueError("You must specify a host by name or IP address")
            return False

        elif self.host is None:
            self.host = host

        if port is not None:
            self.port = port

        self.commPort.connect("tcp://%s:%d" % (self.host, self.port))
        self.dataPort.connect("tcp://%s:%d" % (self.host, self.port + 1))

        self._command(xcaldaq_commands.primary_comm['TESTCOMM'], 0, 0, 0)
        self.setp('CLIENTTYPE', 0, 2)
        print("Testing comm pipe...done!")

        # Find out the geometry, because we basically always want to know this
        self.nchan = self.get('CHANNELS', 0)
        self.ncol = self.get('BOARDS', 0)
        self.nrow = self.nchan / (2 * self.ncol)
        self.npixels = self.ncol * self.nrow
        self.sample_rate = self.get('SAMPLERATE', 0)
        self.num_of_samples = self.get('SAMPLES', 0)
        self.lsync = int(round(self.clockhz / self.sample_rate / float(self.nrow)))
        print(('Connected to server @ %s:%d' % (self.host, self.port)))
        self.print_server_info()

        # print "Testing data pipe...",
        # self._command(xcaldaq_commands.primary_comm['TESTDATA'],0,0,64)
        # _reply = self.dataPort.recv(64)
        # print "done!"
        # for i in range(self.nchan):
        #     self.setp('ACTIVEFLAG',i,0)  # Turn all channels to "inactive"
        self.connected = True
        return True

    def disconnect_server(self):
        """
        Disconnect from the ndfb_server.
        """
        if not self.connected:
            print('Not currently connected')
            return
        print('Shutting down connection to server...')
        self.commPort.disconnect("tcp://%s:%d" % (self.host, self.port))
        self.dataPort.disconnect("tcp://%s:%d" % (self.host, self.port + 1))
        print('...server interface disconnected.')
        self.connected = False

    def comm_recv(self):
        return self.commPort.recv()

    def _get_raw_packets(self, max_bytes=10000000):
        packets = []
        total_bytes = 0
        # first wait for a message
        
        while total_bytes < max_bytes:
            try: # dont know why things need different blocking types
                if self.noblock: # sweeper needs blocking
                    message = self.dataPort.recv_multipart(zmq.NOBLOCK)
                else: #easy client doesn't like blocking
                    message = self.dataPort.recv_multipart()
                if len(message[0]) > 8:
                    print("Error: packet is in the wrong order")
                    packet = message[0]
                elif len(message[1]) > 8:
                    packet = message[1]
                    # print "Receive packet of size %d from chan %d"%(len(packet), 
                    #     struct.unpack("<i", message[0])[0])
                else:
                    print("Error: packet messages have size (%d,%d)" % (len(message[0]), len(message[1])))
                    continue
                packets.append(packet)
                total_bytes += len(packet)
            except zmq.ZMQError as e:
                # Return packets found so far if receive needed to block.
                if isinstance(e, zmq.error.Again):
                    # print "Received %d packets of total size %d"%(len(packets), total_bytes)
                    return packets
                print("Error not anticipated: ", e)
                return packets
        # print "Received %d packets of total size %d"%(len(packets), total_bytes)
        return packets

    def start_streaming(self):
        if not self.connected: return
        for i in self.stream_channels:
            subscription = struct.pack("<i", i)
            self.dataPort.set(zmq.SUBSCRIBE, subscription)
        self.streaming = True
        self.__subclassCallback('startStreaming')

    def stop_streaming(self):
        if not (self.streaming and self.connected): return
        # print 'Stop streaming'
        for i in self.stream_channels:
            subscription = struct.pack("<i", i)
            self.dataPort.set(zmq.UNSUBSCRIBE, subscription)

        self.__subclassCallback('stopStreaming')
        self.streaming = False




class TCPClient(NDFBClient):
    """
    An object to serve as the client of a NASA ndfb_server.

    This is a client for servers before 3.3.0, when the connections used
    raw TCP. For 3.3.0+, use the ZMQClient object, or if unsure, use the 
    client.Client() factory function.
    """

    def __init__(self, host=None, port=None, debug=False, clockmhz=50):
        """
        <host>   The host name or IP address for the ndfb_server.
        <port>   The port number to which you'll connect (if None, then self.DEFAULT_PORT is used).

        If <host> is None, then it must be filled in when calling connect(...)
        """
        NDFBClient.__init__(self)
        self.host = host
        if port is None:
            self.port = self.DEFAULT_PORT
        else:
            self.port = int(port)

        self.debug = debug  # prints out additional debugging statements when this is true
        self.commPort = None
        self.dataPort = None
        self.clockhz=clockmhz*1000000

    def __subclassCallback(self, callbackName, *args, **kwargs):
        try:
            command = 'self.%s(*args, **kwargs)' % callbackName
            exec(command)
        except AttributeError:
            pass

    def connect_server(self, host=None, port=None):
        """
        Connect to an ndfb_server at Internet location <host>:<port>.
        If either <host> or <port> is None, the previous value will be used.  See the constructor
        for how the initial "previous" values are set.

        Internally, this method must connect two pipes: one for commands/replies, and the other
        for data.  (Data is effectively one-way, in that the server ignores any packets sent to it
        on the data pipe.)
        """
        if self.connected:
            print('Already connected')
            return

        if host is None and self.host is None:
            raise ValueError("You must specify a host by name or IP address")
            return False

        elif self.host is None:
            self.host = host

        if port is not None:
            self.port = port

        self.commPort = network_pipe.NetworkPipe(self.host, self.port)
        self.dataPort = network_pipe.NetworkPipe(self.host, self.port)

        self.commPort.settimeout(3.0)
        self._command(xcaldaq_commands.primary_comm['TESTCOMM'], 0, 0, 0)
        self.setp('CLIENTTYPE', 0, 2)
        print("Testing comm pipe...done!")

        # Find out the geometry, because we basically always want to know this
        self.nchan = self.get('CHANNELS', 0)
        self.ncol = self.get('BOARDS', 0)
        self.nrow = self.nchan / (2 * self.ncol)
        self.npixels = self.ncol * self.nrow
        self.sample_rate = self.get('SAMPLERATE', 0)
        self.num_of_samples = self.get('SAMPLES', 0)
        self.lsync = int(round(self.clockhz / self.sample_rate / float(self.nrow)))
        print(('Connected to server @ %s:%d' % (self.host, self.port)))
        self.print_server_info()

        print("Testing data pipe...", end=' ')
        self._command(xcaldaq_commands.primary_comm['TESTDATA'], 0, 0, 64)
        _reply = self.dataPort.recv(64)
        print("done!")
        for i in range(self.nchan):
            self.setp('ACTIVEFLAG', i, 0)  # Turn all channels to "inactive"
        self.connected = True
        return True

    def comm_recv(self):
        return self.commPort.recv(16)

    def disconnect_server(self):
        """
        Disconnect from the ndfb_server.
        """
        if not self.connected:
            print('Not currently connected')
            return
        print('Shutting down connection to server...')
        self.commPort = None
        self.dataPort = None
        print('...server interface disconnected.')
        self.connected = False



    def _get_raw_block(self, max_bytes=-1):
        # max_bytes is unused and is only there to make things work the same as ZMQClient
        BLOCKSIZE = 1024 * 1024 * 8
        bytes_read = 0
        block = ""

        if self.dataPort is None:
            return ""

        try:
            self.dataPort.get_lock()
            try:
                block = self.dataPort.recv(BLOCKSIZE)
            except socket.error:
                pass
            bytes_read = len(block)
        finally:
            self.dataPort.release_lock()
        if self.debug is True:
            print('bytes_read %d' % bytes_read)
        if bytes_read <= 0:
            block = ""
        if self.partial_packet is not None:
            block = self.partial_packet + block
#             print "Had to consume a cached partial packet of length", len(self.partial_packet)
            self.partial_packet = None
        return block

    def _process_raw_block_into_packets(self, dataString):
        # Take a block of data as one string.  Parse it
        # into its separate packets.
        packets = []

        lendata = len(dataString)
        if self.debug is True:
            print("len(dataString) %d" % (lendata,))

        packet_index = 0
        while packet_index < lendata:
            # Packet must be at least 8 bytes long, or you can't read its length
            if lendata < packet_index + 8:
                assert self.partial_packet is None
                self.partial_packet = dataString[packet_index:]
#                 print "Had to cache a short packet of length", len(self.partial_packet)
                return packets

            # Read packet length, and if data is long enough, split off that packet.
            _chan, packet_size = struct.unpack("!II", dataString[packet_index:packet_index + 8])
#            print('chan%d , packet_size %d'%(_chan, packet_size))

            if packet_size == 0:
                # not sure why this was happening, but it caused infinite loops,
                # so I just break out of it emulating no data received
                print("_get_raw_data: packet_size = 0, which should never happen")
                return packets

            if lendata < packet_index + packet_size:
                self.partial_packet = dataString[packet_index:]
#                 print "Had to cache a partial packet of length %d "%len(self.partial_packet)
                return packets

            thispacket = dataString[packet_index:packet_index + packet_size]
            packets.append(thispacket)
            packet_index += packet_size

        return packets

    def _get_raw_packets(self,max_bytes=10000000):
        return self._process_raw_block_into_packets(self._get_raw_block(max_bytes))

    def start_streaming(self):
        if not self.connected: return
        self.partial_packet = None  # just to make sure no old data sneaks in
        for i in self.stream_channels:
            self.setp('ACTIVEFLAG', i, 1)
        self.setp('DATAFLAG', 1, 1)
        self.streaming = True
        self.__subclassCallback('startStreaming')

    def stop_streaming(self):
        if not (self.streaming and self.connected): return
        self.setp('DATAFLAG', 1, 0)
        for i in range(self.nchan):
            self.setp('ACTIVEFLAG', i, 0)  # Turn all channels to "inactive"

        # Now clear the partial packet AND any bytes on the TCP socket buffer
        runt = self._get_raw_block()
        if self.debug:
            print("Stopped streaming... had to ignore %d bytes" % len(runt))
        self.__subclassCallback('stopStreaming')
        self.streaming = False

def Client(*args, **kwargs):
    """Factory function to determine the right type of client through a quick
    connect and disconnect."""

    # First try connecting an old-style TCPClient. If it fails, assume ZMQClient works.
    try:
        c = TCPClient(*args, **kwargs)
        c.connect_server()
        c.disconnect_server()
        return c
    except Exception as e:
        c = ZMQClient(*args, **kwargs)
        return c
