'''
Ethernet based instrument class for SCPI instruments
created June 11, 2012 by Frank Schima
'''
import socket
import time
import instrument


class EthernetInstrument(instrument.Instrument):

    DEFAULT_TIMEOUT = 1
    def __init__(self, hostname, tcpPort=5000, udpPort=5001, maxBufferSize=4096, useUDP = True):

        super(EthernetInstrument, self).__init__()

        self.hostname = hostname
        self.tcpPort = tcpPort
        self.udpPort = udpPort
        self.maxBufferSize = maxBufferSize # max buffer size to read at one time
        self.useUDP = useUDP # True for UDP, False for TCP

    def timedAsk(self, message, timeout=None): # allow a longer timeout for the timing function
        '''
        used to determine how long the ask command is taking
        '''
        startTime = time.time()
        data = self.ask(message, timeout=timeout)
        endTime = time.time()

        executionTime = endTime-startTime

        return data, executionTime

    def ask(self, message, timeout=DEFAULT_TIMEOUT, endline='\r\n'):
        # this way you can switch between UDP and TCP easily
        if self.useUDP:
            return self.askUDP(message, timeout, endline=endline)
        else:
            return self.askTCP(message, timeout, endline=endline)

    def askLong(self, message, endSymbol, timeout=DEFAULT_TIMEOUT):
        # this way you can switch between UDP and TCP easily
        if self.useUDP:
            return self.askLongUDP(message, endSymbol, timeout)
        else:
            return self.askLongTCP(message, endSymbol, timeout)

    def askUDP(self, message, timeout = DEFAULT_TIMEOUT, bufferSize = 1024, endline="\r\n'"):
        # based on http://www.mathworks.com/matlabcentral/fileexchange/24418
        # send message

        # to test UDP on the cryocon, under ubuntu do
        # nc -u 686tupac-cryocon 5001
        # then you can type commands and recieve response, if it doesn't work, you won't be able to it via UDP
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind( ('', self.udpPort))
        sock.settimeout(timeout)

        sock.sendto(message+endline, (self.hostname, self.udpPort))


        data = []
        while data == []:
            data, addr = sock.recvfrom(bufferSize)

            sock.sendto(message+'\r\n', (self.hostname, self.udpPort))

        return data.strip()



    def askTCP(self, message, timeout = DEFAULT_TIMEOUT, endline='\r\n'): # takes about 15-30 ms including opening and closing TCP connection for *IDN>
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #make a TCP socket
        # sock.settimeout(timeout)
        sock.connect( (self.hostname, self.tcpPort))
        sock.send(message + endline)
        data = sock.recv(self.maxBufferSize)
        sock.close()

        return data.strip()

    def askLongUDP(self, message, endSymbol, timeout=DEFAULT_TIMEOUT):
        # based on http://www.mathworks.com/matlabcentral/fileexchange/24418
        # send message
        sendSock = socket.socket( socket.AF_INET, socket.SOCK_DGRAM)


        # recieve response
        recSock = socket.socket( socket.AF_INET, socket.SOCK_DGRAM)
        recSock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        recSock.settimeout(timeout)
        recSock.bind( ('', self.udpPort) )

        sendSock.sendto(message+'\r\n', (self.hostname, self.udpPort))# (string, (ip, port))
        sendSock.close()

        while True:
            try:
                data = recSock.recv(self.maxBufferSize)
            except:
                data = []
                break
            if endSymbol in data:
                total_data.append(data[:data.find(endSymbol)])
                break
            total_data.append(data)
            if len(total_data)>1:
                #check if end_of_data was split
                last_pair=total_data[-2]+total_data[-1]
                if endSymbol in last_pair:
                    total_data[-2]=last_pair[:last_pair.find(endSymbol)]
                    total_data.pop()
                    break

        recSock.close()



        return ''

    def askLongTCP(self, message, endSymbol, timeout=DEFAULT_TIMEOUT):
        '''
        for commands that return lots of data, not just one line
        requires endSymbol, something that signifies the end of the data stream
        '''
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #make a TCP socket
        sock.settimeout(timeout)
        sock.connect( (self.hostname, self.tcpPort))
        sock.sendall(message + '\r\n')
        total_data = []
        while True:
            data=sock.recv(self.maxBufferSize)
            if endSymbol in data:
                total_data.append(data[:data.find(endSymbol)])
                break
            total_data.append(data)
            if len(total_data)>1:
                #check if end_of_data was split
                last_pair=total_data[-2]+total_data[-1]
                if endSymbol in last_pair:
                    total_data[-2]=last_pair[:last_pair.find(endSymbol)]
                    total_data.pop()
                    break
        sock.close()

        return ''.join(total_data)


    def send(self, message, timeout=DEFAULT_TIMEOUT):
        data = self.ask(message+';*ESR?', timeout) # this forces it to send the message followed by something that gives a response, so you can't
        # this behavior is recomended on page 125 of the Crycon 44C instructions
        return data




    def identify(self):
        #print('asking device *IDN?')
        data=self.ask('*IDN?')
        #print('done asking device')
        #print('answer: '+data)

        return data
