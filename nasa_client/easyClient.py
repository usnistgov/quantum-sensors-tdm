from . import easyClientNDFB
from . import easyClientDastard


def EasyClient(host='localhost', port=2011, clockmhz=125, setupOnInit=True):
    """ returns either an easyClientNDFB or an easyClientDastard
    tries to connect to dastard first, then returns NDFB on fail """
    if port == 2011:
        dastardBasePort = 5500
    else:
        dastardBasePort = port
    try:
        return easyClientDastard.EasyClientDastard(host, dastardBasePort, setupOnInit)
    except ConnectionRefusedError:
        print("EasyClient falling back to EasyClientNDFB because of ConnectionRefused (dastard not started)")
        return easyClientNDFB.EasyClientNDFB(host,port,clockmhz)
