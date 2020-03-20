#!/usr/bin/env python

'''
Simple example text client of a NASA server.

This client connects to a NASA server and prints several screen dumps of
the summary data channels.

Created on Nov 10, 2011

@author: fowlerj
'''

import time
import optparse
import nasa_client

class example_client(nasa_client.Client):
    """This client will connect to a server's summary channels."""
    
    def __init__(self, host=None, port=None):
        nasa_client.Client.__init__(self, host=host, port=port)
        self.isStreaming = False
        self.stream_channels=(self.CHANNEL_CODE_ERR_SUMMARY,
                              self.CHANNEL_CODE_FB_SUMMARY,
                              )


def main(host='localhost', port='2011', npackets_max=-1,
         dumphead = False):
    client = example_client(host=host, port=port)
    client.connect_server()
    client.start_streaming()
    npack=0
    while npackets_max <= 0 or npack<=npackets_max:
        payloads, headers = client.get_data_packets()
        if len(payloads)==0:
            time.sleep(0.05)
            continue
        for packet_data, h in zip(payloads, headers):
            if dumphead:
                print h
            if h['chan']==client.CHANNEL_CODE_ERR_SUMMARY:
                print 'Error:    ',packet_data
            else:
                print 'Feedback: ',(packet_data/4)
            npack += 1
    client.stop_streaming()
    client.disconnect_server()

if __name__ == '__main__':
    p = optparse.OptionParser()
    p.add_option('-H','--host', action='store', dest='host', type='string',
                 help='Internet host name/address (default=localhost).')
    p.add_option('-p','--port', action='store', dest='port', type='string',
                 help='TCP port to connect to.')
    p.add_option('-n',action='store', dest='n', type='int',
                 help='Number of packets to dump (<1 means go forever).')
    p.add_option('-d','--dump-headers', action='store_true', dest='dumphead',
                 help='Whether to dump packet headers')
    p.set_defaults(host='localhost')
    p.set_defaults(port='2011')
    p.set_defaults(n=100)
    p.set_defaults(dumphead=False)
    opt, args = p.parse_args()
    host = opt.host
    port = opt.port
    main(host=host, port=port, npackets_max=opt.n, dumphead=opt.dumphead)
