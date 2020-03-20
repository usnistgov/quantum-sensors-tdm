from . import easyClientNDFB
from . import easyClientDastard


def EasyClient(host='localhost', port=2011, clockmhz=50):
    """ returns either an easyClientNDFB or an easyClientDastard
    tries to connect to dastard first, then returns NDFB on fail """
    if port == 2011:
        dastardBasePort = 5500
    else:
        dastardBasePort = port
    try:
        return easyClientDastard.EasyClientDastard(host,dastardBasePort)
    except:
        return easyClientNDFB.EasyClientNDFB(host,port,clockmhz)
