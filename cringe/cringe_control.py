'''
Helper for talking to cringe remotely

G Hilton 2020-12
'''
import zmq
import zmq_prevent_hang_hack
import json
import numpy as np

CRINGE_PORT = 5509

CRINGE_COMMANDS = {
    'setup_crate':{'fname':'full_crate_init', 'args':None, 'help':'Full crate init'},
    'full_tune':{'fname': 'extern_tune', 'args':None, 'help':'Full tune'},
    'relock_fba':{'fname':'rpc_relock_fba', 'args': ['col', 'row'], 'help': 'relock FBA for col and row, column matches dastard after tune'},
    'relock_all_locked_fba':{'fname':'rpc_relock_all_locked_fba', 'args': ['col'], 'help': 'relock FBA for each row in col that is locked'},
    'relock_fbb':{'fname':'rpc_relock_fbb', 'args': ['col', 'row'], 'help': 'relock FBB for col and row, column matches dastard after tune'},
    'set_arl_off':{'fname':'rpc_set_arl_off', 'args':None, 'help': 'set ARL (autorelock) off for all rows in this column'},
    'set_fba_offset':{'fname':'rpc_set_fba_offset', 'args':['col', 'fba_offset'], 'help':'set fba_offset for all rows in this column'},
    'set_fb_i':{'fname':'rpc_set_fb_i', 'args':['col', 'fb_i'], 'help':'set feedback parameter I for all rows in this column'},
    'set_fb_p':{'fname':'rpc_set_fb_p', 'args':['col', 'fb_p'], 'help':'set feedback parameter P for all rows in this column'},
    'set_arl_params':{'fname':'rpc_set_arl_params', 'args':['flux_jump_threshold_dac_units', 'plus_event_reset_delay_frm_units', 'minus_event_reset_delay_frm_units'], 'help':'set all the arl params'},
    'set_tower_channel':{'fname':'rpc_set_tower_channel', 'args':['cardname', 'bayname', 'dacvalue'], 'help':'set the dac value for a tower card by cardname and bayname (strings)'},
    'set_tower_card_all_channels':{'fname':'rpc_set_tower_card_all_channels', 'args':['cardname', 'dacvalue'], 'help':'set the dac value for all tower channels in one card'},
    'get_fba_offsets':{'fname':'rpc_get_fba_offsets', 'args':[], 'help': "returns fba offsets as an array"},
    'set_clk_fba_dc':{'fname':'rpc_set_clk_fba_dc', "args":['dac'], 'help':'set '}
#    'devtest':{'fname':'devtest', 'args':['arg1'], 'help':'temporary for development testing'},      
#    'cmd':{'fname':name, 'args':[], 'help':''},      
    }


def build_zmq_addr(host='localhost', port=CRINGE_PORT):
    return 'tcp://%s:%d' % (host,port)

class CringeControl:
    ''' class for your script to send remote commands to cringe'''
    def __init__(self, host='localhost'):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.LINGER = 3000 # ms
        self.socket.connect(build_zmq_addr(host))

    def send(self, message):
        self.socket.send(message.encode('ascii', 'ignore'))
        reply = self.socket.recv().decode()
        return reply

    def setup_crate(self):
        ''' This command was in cringe before I added this module'''
        command = 'setup_crate'
        reply = self.send(command)
        return reply

    def full_tune(self):
        ''' This command was in cringe before I added this module'''
        command = 'full_tune'
        reply = self.send(command)
        return reply

    def set_tower_channel(self, cardname, bayname, dacvalue):
        return self.send(' '.join(('set_tower_channel', cardname, bayname, str(int(dacvalue)))))

    def set_tower_card_all_channels(self, cardname, dacvalue):
        return self.send(' '.join(('set_tower_card_all_channels', cardname, dacvalue)))

    def relock_fba(self, col, row):
        return self.send(' '.join(('relock_fba', str(int(col)), str(int(row)))))

    def relock_fbb(self, col, row):
        return self.send(' '.join(('relock_fbb', str(int(col)), str(int(row)))))

    def set_arl_off(self, col):
        return self.send(" ".join(('set_arl_off', str(int(col)))))

    def set_arl_params(self, flux_jump_threshold_dac_units,
    plus_event_reset_delay_frm_units, minus_event_reset_delay_frm_units):
        return self.send(" ".join(("set_arl_params", str(int(flux_jump_threshold_dac_units)),
        str(int(plus_event_reset_delay_frm_units)), str(int(minus_event_reset_delay_frm_units)))))

    def set_fba_offset(self, col, fba_offset):
        return self.send(" ".join(("set_fba_offset", str(int(col)), str(int(fba_offset)))))

    def set_fb_i(self, col, fb_i):
        return self.send(" ".join(("set_fb_i", str(int(col)), str(int(fb_i)))))

    def set_fb_p(self, col, fb_p):
        return self.send(" ".join(("set_fb_i", str(int(col)), str(int(fb_p)))))

    def relock_all_locked_fba(self, col):
        return self.send(" ".join(("relock_all_locked_fba", str(int(col)))))

    def get_fba_offsets(self):
        reply = self.send("".join(("get_fba_offsets")))
        if not reply.startswith("ok: "):
            return reply
        a,b,c = reply[5:].split(",",2)
        ncol = int(a)
        nrow = int(b)
        vals = json.loads(c[:-1])
        vals2 = np.array(vals)
        vals3 = np.reshape(vals2,(8,-1))
        return reply, vals3

    def set_clk_fba_dc(self, dac):
        return self.send(" ".join(("set_clk_fba_dc", str(int(dac)))))
  


    def test(self):
        command = 'devtest'
        reply = self.send(command)
        return reply

def cc_help():
    help_text = []
    for cmd in CRINGE_COMMANDS:
        help_text.append('%s : %s' % (cmd, CRINGE_COMMANDS[cmd]['help']))
    return '\n'.join(help_text)


def cringe_control_commandline_main():
    ''' a commandline client for playing with this'''
    cc = CringeControl()
    print('Talk do cringe - enter a command, quit or help')
    while True:
        request_text = input('?: ')
        if request_text.lower().strip() == 'quit':
            break
        elif request_text.lower().strip() == 'help':
            print(cc_help())
            continue
        request_words = request_text.split()
        if request_words[0].lower() not in  CRINGE_COMMANDS:
            print('Command %s not found' % request_words[0])
            print(cc_help())
            continue
        result = cc.send(request_text)
        print('   Reply: %s' % result)


if __name__ == '__main__':
    cringe_control_commandline_main()