
comm_ack = {
        'COMMAND':0,
        'ACKNOWLEDGE':1,
        }

primary_comm = {
        'GET':0,
        'SET':1,
        'KILLME':2,
        'KEEPALIVE':3,
        'TESTCOMM':4,
        'TESTDATA':5,
        'REWIND_DATA_FILE':6,
        }

secondary_comm = {
    # Per-client, not channel-specific items:
        # Boolean parameters
        'DATAFLAG':0,
        'INITIALIZED':1,
        
        # Long integer parameters
        'VERSION':49,
        'BUFFERLEVEL':50,
        'CHANNELS':51,
        'STARTTS':52,
        'STARTTUS':53,
        'SAMPLERATE':54,
        'BOARDS':55,
        'SAMPLES':56,
        'SERVERTYPE':57,
        'STREAMS_PER_BOARD':58,
        'REDECIMATE':59,
        'CLIENTTYPE':60,
        
        
    # Channel-specific items
        # Boolean parameters
        'ACTIVEFLAG':100,
        'MIXFLAG':101,
        'DECIMATEFLAG':102,
        'MIXINVERSIONFLAG':103,
        'DECIMATEAVGFLAG':104,
        
        # Short integer parameters
        'DECIMATELEVEL':150,
        
        # Long integer parameters
        'MIXLEVEL': 200,
        'GAINLEVEL':201,
        'RAW_MIN_LEVEL':202,
        'RAW_MAX_LEVEL':203,
        }
